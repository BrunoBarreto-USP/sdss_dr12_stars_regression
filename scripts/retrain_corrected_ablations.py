"""Matched-head ablations of trial 099, preserving all earlier experiments."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.train_paper_hyperparams import build_paper_fixed_model, FixedPaperHP
from training.train_best_tuner_trial import FixedTrialHP
from training.training import _create_dataset, NormalizedMaeSumCallback
from data_preprocessing.augmentation import (
    augment_training_data,
    prepare_multitask_dicts,
)
from data_preprocessing.validation import validate_compact_dataset
from model_definitions.model_definitions import _build_task_output, WarmupCosineDecay
from results_and_evaluations.measure_madds import estimate_model_flops

KEYS = ("teff_output", "feh_output", "logg_output")
TARGETS = ("Teff", "[Fe/H]", "log g")
VARIANTS = (
    "no_residual_no_layernorm",
    "single_teff",
    "single_feh",
    "single_logg",
    "no_skip",
    "no_layernorm",
    "no_noise_augmentation",
)


def cached_augmentation(x, path):
    """Write the same offline noisy copies in chunks, avoiding several large copies in RAM."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError(
            "Cached augmentation requires nonempty finite 2-D training inputs."
        )
    n = len(x)
    rng = np.random.default_rng(42)
    ready = path.with_suffix(".ready.json")
    create = not path.exists() or not ready.exists()
    fingerprint = hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()
    metadata = json.loads(ready.read_text()) if not create else {}
    expected_shape = [3 * n, x.shape[1]]
    if not create and (
        metadata.get("shape") != expected_shape
        or metadata.get("seed") != 42
        or metadata.get("factor") != 3
    ):
        raise ValueError("Augmentation cache configuration does not match this run.")
    if metadata.get("source_sha256", fingerprint) != fingerprint:
        raise ValueError("Augmentation cache belongs to different training inputs.")
    validate_legacy = not create and "source_sha256" not in metadata
    augmented = np.lib.format.open_memmap(
        path, mode="w+" if create else "r", dtype=np.float32, shape=(3 * n, x.shape[1])
    )
    if tuple(augmented.shape) != tuple(expected_shape) or augmented.dtype != np.float32:
        raise ValueError("Augmentation cache has unexpected shape or dtype.")
    if create:
        augmented[:n] = x
    elif validate_legacy:
        np.testing.assert_array_equal(augmented[:n], x)
    std = x.std(axis=1, keepdims=True)
    for copy in (1, 2):
        level = rng.uniform(0.01, 0.05)
        for start in range(0, n, 256):
            stop = min(start + 256, n)
            noise = rng.standard_normal((stop - start, x.shape[1])).astype(np.float32)
            expected = x[start:stop] + level * std[start:stop] * noise
            if create:
                augmented[copy * n + start : copy * n + stop] = expected
            elif validate_legacy:
                np.testing.assert_array_equal(
                    augmented[copy * n + start : copy * n + stop], expected
                )
    order = rng.permutation(3 * n)
    if create:
        augmented.flush()
    if create or validate_legacy:
        ready.write_text(
            json.dumps(
                {
                    "shape": list(augmented.shape),
                    "seed": 42,
                    "factor": 3,
                    "noise_fraction": [0.01, 0.05],
                    "source_sha256": fingerprint,
                }
            )
        )
    return augmented, order


def indexed_dataset(x, labels, order, keys, *, repeat_original=False, batch_size=256):
    targets = {k: np.tile(labels[k], 3)[order] for k in keys}
    dataset = tf.data.Dataset.from_tensor_slices((order, targets))
    dataset = dataset.shuffle(buffer_size=min(len(order), 50000)).batch(
        batch_size, drop_remainder=True
    )

    def fetch(indices, targets):
        features = tf.numpy_function(
            lambda ids: np.asarray(
                x[ids % len(x) if repeat_original else ids], dtype=np.float32
            ),
            [indices],
            tf.float32,
        )
        features.set_shape((None, x.shape[1]))
        return features, targets

    return dataset.map(fetch, num_parallel_calls=1).prefetch(1)


class AliasSingleMetric(tf.keras.callbacks.Callback):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def on_epoch_end(self, epoch, logs=None):
        for prefix in ("", "val_"):
            if prefix + "mae" in logs:
                logs[prefix + self.key + "_mae"] = logs[prefix + "mae"]


def build_variant(name, epochs, steps, weights, *, batch_size=256):
    if name == "no_noise_augmentation":
        return build_paper_fixed_model(
            input_dim=4000,
            total_steps=steps * epochs,
            loss_weights=weights,
            batch_size=batch_size,
            jit_compile=False,
        ), list(range(3))
    if name.startswith("single_"):
        index = VARIANTS.index(name) - 1
        full = build_paper_fixed_model(
            input_dim=4000,
            total_steps=steps * epochs,
            loss_weights=weights,
            batch_size=batch_size,
            jit_compile=False,
        )
        return tf.keras.Model(
            full.input, {KEYS[index]: full.get_layer(KEYS[index]).output}
        ), [index]
    hp = FixedPaperHP.values
    use_norm = name not in ("no_layernorm", "no_residual_no_layernorm")
    use_skip = name not in ("no_skip", "no_residual_no_layernorm")
    assert (
        hp["shared_blocks"] == 1 and hp["initial_dropout"] == hp["trunk_dropout"] == 0
    )
    inputs = tf.keras.layers.Input(shape=(4000,))
    x = tf.keras.layers.Dense(hp["initial_units"], name="input_stem_dense")(inputs)
    if use_norm:
        x = tf.keras.layers.LayerNormalization(name="input_stem_ln")(x)
    x = tf.keras.layers.Activation("gelu", name="input_stem_activation")(x)
    shortcut = x
    if use_norm:
        x = tf.keras.layers.LayerNormalization(name="shared_block_0_ln_0")(x)
    x = tf.keras.layers.Dense(hp["block_units"], name="shared_block_0_dense_0")(x)
    x = tf.keras.layers.Activation("gelu", name="shared_block_0_activation_0")(x)
    if use_norm:
        x = tf.keras.layers.LayerNormalization(name="shared_block_0_ln_1")(x)
    x = tf.keras.layers.Dense(hp["block_units"], name="shared_block_0_dense_1")(x)
    if use_skip:
        if use_norm:
            shortcut = tf.keras.layers.LayerNormalization(
                name="shared_block_0_skip_ln"
            )(shortcut)
        shortcut = tf.keras.layers.Dense(
            hp["block_units"], name="shared_block_0_skip_dense"
        )(shortcut)
        x = tf.keras.layers.Add(name="shared_block_0_add")([x, shortcut])
    if use_norm:
        x = tf.keras.layers.LayerNormalization(name="trunk_final_ln")(x)
    outputs = [
        _build_task_output(
            x, hp["block_units"], key, i, hp=FixedTrialHP(hp), dropout_rate=0.0
        )
        for i, key in enumerate(KEYS)
    ]
    model = tf.keras.Model(inputs, outputs)
    assert (
        any(
            isinstance(layer, tf.keras.layers.LayerNormalization)
            for layer in model.layers
        )
        == use_norm
    )
    assert (
        any(isinstance(layer, tf.keras.layers.Add) for layer in model.layers)
        == use_skip
    )
    expected = {
        "no_residual_no_layernorm": 525251,
        "no_skip": 525891,
        "no_layernorm": 529379,
    }
    assert model.count_params() == expected[name], model.count_params()
    return model, list(range(3))


def main():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from stellar_workflow.compat import legacy_main
    return legacy_main("ablations")


if __name__ == "__main__":
    main()
