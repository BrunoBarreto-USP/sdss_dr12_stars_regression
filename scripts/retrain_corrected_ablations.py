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


def indexed_dataset(x, labels, order, keys, *, repeat_original=False):
    targets = {k: np.tile(labels[k], 3)[order] for k in keys}
    dataset = tf.data.Dataset.from_tensor_slices((order, targets))
    dataset = dataset.shuffle(buffer_size=min(len(order), 50000)).batch(
        256, drop_remainder=True
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


def build_variant(name, epochs, steps, weights):
    if name == "no_noise_augmentation":
        return build_paper_fixed_model(
            input_dim=4000,
            total_steps=steps * epochs,
            loss_weights=weights,
            batch_size=256,
            jit_compile=False,
        ), list(range(3))
    if name.startswith("single_"):
        index = VARIANTS.index(name) - 1
        full = build_paper_fixed_model(
            input_dim=4000,
            total_steps=steps * epochs,
            loss_weights=weights,
            batch_size=256,
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


def summarize(base):
    reports = {
        name: json.loads((base / name / "metrics.json").read_text())
        for name in VARIANTS
        if (base / name / "metrics.json").exists()
    }
    missing = set(VARIANTS[:4]) - reports.keys()
    if missing:
        raise FileNotFoundError(f"Incomplete original ablations: {sorted(missing)}")
    rows = []
    for variant, items in (
        ("no_residual_no_layernorm", [reports[VARIANTS[0]]]),
        ("no_multitask", [reports[k] for k in VARIANTS[1:4]]),
    ):
        scores = {
            key: value for item in items for key, value in item["test_mae"].items()
        }
        rows.append(
            {
                "variant": variant,
                "params": sum(r["params"] for r in items),
                "flops": sum(r["flops"] for r in items),
                **dict(
                    zip(
                        ("MAE_Teff", "MAE_[Fe/H]", "MAE_log g"),
                        [scores[t] for t in TARGETS],
                    )
                ),
            }
        )
    for name in VARIANTS[4:]:
        if name in reports:
            item = reports[name]
            rows.append(
                {
                    "variant": name,
                    "params": item["params"],
                    "flops": item["flops"],
                    **dict(
                        zip(
                            ("MAE_Teff", "MAE_[Fe/H]", "MAE_log g"),
                            [item["test_mae"][t] for t in TARGETS],
                        )
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(base / "ablation_summary.csv", index=False)
    if set(reports) == set(VARIANTS):
        from results_and_evaluations.paired_bootstrap_all_models import (
            bootstrap_paired_differences,
        )

        reference = np.load(base.parent / "test_predictions.npz")
        errors = abs(reference["y_pred"] - reference["y_true"])
        comparators = {}
        for name in (VARIANTS[0], *VARIANTS[4:]):
            p = np.load(base / name / "test_predictions.npz")
            np.testing.assert_array_equal(p["test_ids"], reference["test_ids"])
            np.testing.assert_array_equal(p["y_true"], reference["y_true"])
            comparators[name] = abs(p["y_pred"] - p["y_true"]) - errors
        single = np.column_stack(
            [
                np.load(base / name / "test_predictions.npz")["y_pred"][:, 0]
                for name in VARIANTS[1:4]
            ]
        )
        comparators["no_multitask"] = abs(single - reference["y_true"]) - errors
        names = list(comparators)
        difference, ci = bootstrap_paired_differences(
            np.stack([comparators[n] for n in names], axis=2)
        )
        paired = {
            "n_test": len(errors),
            "n_boot": 10000,
            "seed": 42,
            "difference_definition": "MAE_ablation minus MAE_full_model; positive favors full model",
            "intervals": "individual paired percentile 95%; fixed checkpoints, unadjusted for multiplicity",
            "comparisons": {
                name: {
                    target: {
                        "difference": float(difference[j, k]),
                        "ci_95": ci[:, j, k].tolist(),
                    }
                    for j, target in enumerate(TARGETS)
                }
                for k, name in enumerate(names)
            },
        }
        (base / "ablation_paired_bootstrap.json").write_text(
            json.dumps(paired, indent=2), encoding="utf-8"
        )
    print("Matched-head ablations:", rows, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "outputs/paper_figures/single_column/corrected_best_trial",
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Shuffle indices into memory-mapped copies of the same augmented spectra.",
    )
    args = parser.parse_args()
    if args.threads < 1 or len(set(args.variants)) != len(args.variants):
        parser.error("Thread count must be positive and variants must not be repeated.")
    if (
        "no_noise_augmentation" in args.variants
        and not args.low_memory
        and not args.summarize_only
    ):
        parser.error("The matched-exposure no-noise control requires --low-memory.")
    base = args.run_dir / "ablations"
    base.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(base)
        return
    for name in args.variants:
        if (base / name / "history.csv").exists():
            raise FileExistsError(f"Preserve existing training: {base / name}")
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.config.experimental.enable_op_determinism()
    data = np.load(ROOT / "data/sdss_dr12_processed_flux_benchmark.npz")
    validate_compact_dataset(data, ROOT / "data")
    reference = np.load(ROOT / "data/paper_hyperparams_test_predictions.npz")
    np.testing.assert_array_equal(reference["test_ids"], data["test_ids"])
    x = data["X_train_features"]
    xv = data["X_val_features"]
    yt, yv = prepare_multitask_dicts(data["y_train_targets"], data["y_val_targets"])
    if args.low_memory:
        xa, order = cached_augmentation(x, base / "augmented_train.npy")
    else:
        xa, yt = augment_training_data(x, yt, aug_factor=3, noise_level=0.05, seed=42)
    config = json.loads((base.parent / "run_config.json").read_text())
    if config["seed"] != 42 or config["epochs"] != 80 or config["batch_size"] != 256:
        raise ValueError(
            "These controlled ablations require the 80-epoch, batch-256, seed-42 reference protocol."
        )
    weights = config["loss_weights"]
    std = config["target_standard_deviations"]
    steps = len(xa) // 256
    for name in args.variants:
        out = base / name
        out.mkdir(exist_ok=True)
        if (out / "history.csv").exists():
            raise FileExistsError(out)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        model, indices = build_variant(name, 80, steps, weights)
        keys = [KEYS[i] for i in indices]
        for i in indices:
            actual = [
                l.units
                for l in model.layers
                if l.name.startswith(KEYS[i] + "_head_dense_")
            ]
            assert actual == config["heads"][TARGETS[i]], actual
        schedule = WarmupCosineDecay(
            0.005,
            total_steps=steps * 80,
            warmup_steps=int(steps * 80 * 0.05),
            alpha=0.05,
        )
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=schedule, weight_decay=1e-5, global_clipnorm=5
            ),
            loss={k: tf.keras.losses.Huber(delta=1.0) for k in keys},
            loss_weights={k: (weights[k] if len(keys) > 1 else 1.0) for k in keys},
            metrics={k: [tf.keras.metrics.MeanAbsoluteError(name="mae")] for k in keys},
        )
        train = (
            indexed_dataset(
                x if name == "no_noise_augmentation" else xa,
                yt,
                order,
                keys,
                repeat_original=name == "no_noise_augmentation",
            )
            if args.low_memory
            else _create_dataset(
                xa, {k: yt[k] for k in keys}, batch_size=256, shuffle=True, cache=False
            )
        )
        validation = _create_dataset(
            xv, {k: yv[k] for k in keys}, batch_size=256, shuffle=False, cache=False
        )
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        train = train.with_options(options)
        validation = validation.with_options(options)
        callbacks = ([AliasSingleMetric(keys[0])] if len(keys) == 1 else []) + [
            NormalizedMaeSumCallback(std, keys),
            tf.keras.callbacks.ModelCheckpoint(
                str(out / "best.weights.h5"),
                monitor="val_normalized_mae_sum",
                mode="min",
                save_best_only=True,
                save_weights_only=True,
            ),
            tf.keras.callbacks.CSVLogger(str(out / "history.csv")),
        ]
        print("Starting matched-head variant:", name, model.count_params(), flush=True)
        history = model.fit(
            train, validation_data=validation, epochs=80, callbacks=callbacks, verbose=2
        )
        model.load_weights(out / "best.weights.h5")
        model.save(out / "model.keras")
        raw = model.predict(data["X_test_features"], batch_size=256, verbose=0)
        if isinstance(raw, dict):
            raw = np.column_stack([np.asarray(raw[k]).ravel() for k in keys])
        elif isinstance(raw, list):
            raw = np.column_stack([np.asarray(p).ravel() for p in raw])
        else:
            raw = np.asarray(raw).reshape(-1, 1)
        pred = (
            raw.astype(np.float64) * data["label_robust_scale"][indices]
            + data["label_robust_center"][indices]
        )
        true = reference["y_true"][:, indices]
        np.savez_compressed(
            out / "test_predictions.npz",
            test_ids=reference["test_ids"],
            y_true=true,
            y_pred=pred,
            target_names=np.asarray(TARGETS)[indices],
        )
        params, flops, macs = estimate_model_flops(out / "model.keras", 4000)
        report = {
            "variant": name,
            "params": params,
            "flops": flops,
            "epochs": 80,
            "seed": 42,
            "selected_epoch": int(np.argmin(history.history["val_normalized_mae_sum"]))
            + 1,
            "test_mae": dict(
                zip(
                    [TARGETS[i] for i in indices],
                    np.mean(abs(pred - true), axis=0).tolist(),
                )
            ),
            "heads": {TARGETS[i]: config["heads"][TARGETS[i]] for i in indices},
            "source_trial_id": config["source_trial_id"],
        }
        report["training_data_storage"] = (
            "memory-mapped identical offline copies"
            if args.low_memory
            else "in-memory offline copies"
        )
        report.update(
            {
                "batch_size": 256,
                "steps_per_epoch": steps,
                "training_rows_per_epoch": len(xa),
                "original_training_objects": len(x),
                "loss_weights": {
                    k: (weights[k] if len(keys) > 1 else 1.0) for k in keys
                },
                "noise_augmentation": name != "no_noise_augmentation",
                "sample_exposure": "three unchanged copies, same shuffle order and optimizer-step budget"
                if name == "no_noise_augmentation"
                else "one original and two fixed noisy copies",
            }
        )
        (out / "metrics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print("Finished matched-head variant:", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
