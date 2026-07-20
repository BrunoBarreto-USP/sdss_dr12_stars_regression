"""Run architectural ablations for this repository's fixed paper model.

The implementation mirrors ``training.train_paper_hyperparams`` rather than
the separate master-IC experiment: it retains this project's selected widths,
dropout, optimizer, target scaling, augmentation, and validation protocol.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == SCRIPT_DIR:
    sys.path.pop(0)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.augmentation import augment_training_data, prepare_multitask_dicts
from model_definitions.model_definitions import (
    DEFAULT_COSINE_ALPHA,
    DEFAULT_HUBER_DELTA,
    WARMUP_FRACTION,
    WarmupCosineDecay,
)
from training.train_paper_hyperparams import FixedPaperHP
from training.training import (
    NORMALIZED_MAE_SUM_NAME,
    NormalizedMaeSumCallback,
    _compute_target_standard_deviations,
    _create_dataset,
    cast_target_dicts_to_float32,
    compute_inverse_variance_loss_weights,
    configure_cpu_runtime,
    configure_gpu_runtime,
)

DEFAULT_DATA = PROJECT_ROOT / "data" / "sdss_dr12_processed_flux_benchmark.npz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results_and_evaluations" / "paper_ablations"
TARGET_KEYS = ("teff_output", "feh_output", "logg_output")
TARGET_LABELS = ("Teff", "[Fe/H]", "log g")
DEFAULT_VARIANTS = ("no_residual", "no_residual_no_layernorm", "no_multitask")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _target_scalers_from_npz(data) -> list[RobustScaler]:
    scalers: list[RobustScaler] = []
    for center, scale in zip(data["label_robust_center"], data["label_robust_scale"]):
        scaler = RobustScaler()
        scaler.center_ = np.asarray([center], dtype=np.float64)
        scaler.scale_ = np.asarray([scale], dtype=np.float64)
        scaler.n_features_in_ = 1
        scalers.append(scaler)
    return scalers


def _inverse_targets(values: np.ndarray, scalers: Sequence[RobustScaler]) -> np.ndarray:
    unscaled = np.empty_like(values, dtype=np.float64)
    for index, scaler in enumerate(scalers):
        unscaled[:, index] = scaler.inverse_transform(values[:, [index]]).ravel()
    return unscaled


def _physical_mae(predictions: np.ndarray, targets: np.ndarray, scalers: Sequence[RobustScaler]) -> dict[str, float]:
    predictions = _inverse_targets(predictions, scalers)
    targets = _inverse_targets(targets, scalers)
    return {
        label: float(np.mean(np.abs(predictions[:, index] - targets[:, index])))
        for index, label in enumerate(TARGET_LABELS)
    }


def _parse_variants(raw_variants: str) -> list[str]:
    variants = [variant.strip() for variant in raw_variants.split(",") if variant.strip()]
    unknown = sorted(set(variants) - set(DEFAULT_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown paper-model ablations: {unknown}")
    if not variants:
        raise ValueError("At least one ablation variant is required.")
    return variants


def _dense_head(x, *, output_name: str, units: tuple[int, ...]):
    for index, width in enumerate(units):
        x = tf.keras.layers.Dense(width, activation="gelu", name=f"{output_name}_head_dense_{index}")(x)
    return tf.keras.layers.Dense(1, activation="linear", dtype="float32", name=output_name)(x)


def build_paper_ablation_model(
    *,
    input_dim: int,
    total_steps: int,
    batch_size: int,
    loss_weights: dict[str, float] | None,
    use_residual: bool,
    use_layernorm: bool,
    output_name: str | None = None,
    jit_compile: bool = False,
) -> tf.keras.Model:
    """Build a paper-model ablation while retaining all non-ablated choices."""
    hp = FixedPaperHP.values
    inputs = tf.keras.layers.Input(shape=(input_dim,), name="stellar_input")

    x = tf.keras.layers.Dense(hp["initial_units"], name="input_stem_dense")(inputs)
    if use_layernorm:
        x = tf.keras.layers.LayerNormalization(name="input_stem_ln")(x)
    x = tf.keras.layers.Activation("gelu", name="input_stem_activation")(x)

    shortcut = x
    if use_layernorm:
        x = tf.keras.layers.LayerNormalization(name="shared_block_0_ln_0")(x)
    x = tf.keras.layers.Dense(hp["block_units"], name="shared_block_0_dense_0")(x)
    x = tf.keras.layers.Activation("gelu", name="shared_block_0_activation_0")(x)
    if use_layernorm:
        x = tf.keras.layers.LayerNormalization(name="shared_block_0_ln_1")(x)
    x = tf.keras.layers.Dense(hp["block_units"], name="shared_block_0_dense_1")(x)
    x = tf.keras.layers.Dropout(hp["trunk_dropout"], name="shared_block_0_dropout")(x)

    if use_residual:
        if use_layernorm:
            shortcut = tf.keras.layers.LayerNormalization(name="shared_block_0_shortcut_ln")(shortcut)
        shortcut = tf.keras.layers.Dense(hp["block_units"], name="shared_block_0_shortcut_dense")(shortcut)
        x = tf.keras.layers.Add(name="shared_block_0_add")([x, shortcut])
    if use_layernorm:
        x = tf.keras.layers.LayerNormalization(name="trunk_final_ln")(x)

    head_units = {
        "teff_output": (48, 32),
        "feh_output": (16,),
        "logg_output": (48,),
    }
    selected_outputs = TARGET_KEYS if output_name is None else (output_name,)
    outputs = [_dense_head(x, output_name=name, units=head_units[name]) for name in selected_outputs]

    warmup_steps = int(total_steps * WARMUP_FRACTION)
    learning_rate = WarmupCosineDecay(
        hp["learning_rate"] * (batch_size / 256) ** 0.5,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        alpha=DEFAULT_COSINE_ALPHA,
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=hp["weight_decay"],
        global_clipnorm=5.0,
    )

    if output_name is None:
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="paper_multitask_ablation")
        model.compile(
            optimizer=optimizer,
            loss={name: tf.keras.losses.Huber(delta=DEFAULT_HUBER_DELTA) for name in TARGET_KEYS},
            loss_weights=loss_weights,
            metrics={name: [tf.keras.metrics.MeanAbsoluteError(name="mae")] for name in TARGET_KEYS},
            jit_compile=jit_compile,
        )
        return model

    model = tf.keras.Model(inputs=inputs, outputs=outputs[0], name=f"{output_name}_paper_single_task_ablation")
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=DEFAULT_HUBER_DELTA),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        jit_compile=jit_compile,
    )
    return model


def _predict_multitask(model: tf.keras.Model, features: np.ndarray, *, batch_size: int) -> np.ndarray:
    predictions = model.predict(features, batch_size=batch_size, verbose=0)
    return np.column_stack([np.asarray(prediction).reshape(-1) for prediction in predictions])


def _create_single_target_dataset(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    cache: bool,
) -> tf.data.Dataset:
    """Match the paper-model data pipeline for a single scalar target."""
    dataset = tf.data.Dataset.from_tensor_slices(
        (np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)),
    )
    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(features), 50_000))
    return dataset.batch(batch_size, drop_remainder=shuffle).prefetch(tf.data.AUTOTUNE)


def _train_multitask_variant(
    *,
    variant: str,
    use_residual: bool,
    use_layernorm: bool,
    x_train: np.ndarray,
    y_train: dict[str, np.ndarray],
    x_val: np.ndarray,
    y_val: dict[str, np.ndarray],
    x_test: np.ndarray,
    target_standard_deviations: dict[str, float],
    loss_weights: dict[str, float],
    args: argparse.Namespace,
) -> tuple[np.ndarray, int]:
    steps_per_epoch = max(1, len(x_train) // args.batch_size)
    model = build_paper_ablation_model(
        input_dim=x_train.shape[1],
        total_steps=steps_per_epoch * args.epochs,
        batch_size=args.batch_size,
        loss_weights=loss_weights,
        use_residual=use_residual,
        use_layernorm=use_layernorm,
        jit_compile=args.use_xla,
    )
    checkpoint = args.output_dir / "_checkpoints" / f"{variant}.weights.h5"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        _create_dataset(x_train, y_train, args.batch_size, shuffle=True, cache=args.cache),
        validation_data=_create_dataset(x_val, y_val, args.batch_size, shuffle=False, cache=args.cache),
        epochs=args.epochs,
        callbacks=[
            NormalizedMaeSumCallback(target_standard_deviations, TARGET_KEYS),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint,
                monitor=f"val_{NORMALIZED_MAE_SUM_NAME}",
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
        ],
        verbose=1,
    )
    model.load_weights(checkpoint)
    checkpoint.unlink(missing_ok=True)
    if args.save_models:
        model_dir = args.output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_dir / f"{variant}.keras")
    return _predict_multitask(model, x_test, batch_size=args.batch_size), int(model.count_params())


def _train_no_multitask_variant(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int]:
    predictions = np.empty((len(x_test), len(TARGET_KEYS)), dtype=np.float32)
    total_params = 0
    for target_index, output_name in enumerate(TARGET_KEYS):
        steps_per_epoch = max(1, len(x_train) // args.batch_size)
        model = build_paper_ablation_model(
            input_dim=x_train.shape[1],
            total_steps=steps_per_epoch * args.epochs,
            batch_size=args.batch_size,
            loss_weights=None,
            use_residual=True,
            use_layernorm=True,
            output_name=output_name,
            jit_compile=args.use_xla,
        )
        checkpoint = args.output_dir / "_checkpoints" / f"no_multitask_{output_name}.weights.h5"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.fit(
            _create_single_target_dataset(
                x_train,
                y_train[:, target_index],
                args.batch_size,
                shuffle=True,
                cache=args.cache,
            ),
            validation_data=_create_single_target_dataset(
                x_val,
                y_val[:, target_index],
                args.batch_size,
                shuffle=False,
                cache=args.cache,
            ),
            epochs=args.epochs,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint,
                    monitor="val_mae",
                    mode="min",
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1,
                ),
            ],
            verbose=1,
        )
        model.load_weights(checkpoint)
        checkpoint.unlink(missing_ok=True)
        predictions[:, target_index] = model.predict(x_test, batch_size=args.batch_size, verbose=0).reshape(-1)
        total_params += int(model.count_params())
        if args.save_models:
            model_dir = args.output_dir / "models" / "no_multitask"
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(model_dir / f"{output_name}.keras")
    return predictions, total_params


def run_paper_ablations(args: argparse.Namespace) -> pd.DataFrame:
    """Train requested ablations and save unscaled test metrics."""
    args.output_dir = Path(args.output_dir)
    variants = _parse_variants(args.variants)
    runtime = configure_cpu_runtime() if args.cpu else configure_gpu_runtime(
        require_gpu=False,
        enable_mixed_precision=not args.no_mixed_precision,
    )
    _set_seed(args.seed)
    print("Runtime:", runtime)
    print("Ablations:", ", ".join(variants))

    data = np.load(args.data, allow_pickle=True)
    x_train = np.asarray(data["X_train_features"], dtype=np.float32)
    x_val = np.asarray(data["X_val_features"], dtype=np.float32)
    x_test = np.asarray(data["X_test_features"], dtype=np.float32)
    y_train = np.asarray(data["y_train_targets"], dtype=np.float32)
    y_val = np.asarray(data["y_val_targets"], dtype=np.float32)
    y_test = np.asarray(data["y_test_targets"], dtype=np.float32)

    y_train_dict, y_val_dict = prepare_multitask_dicts(y_train, y_val)
    y_train_dict, y_val_dict = cast_target_dicts_to_float32(y_train_dict, y_val_dict)
    if args.aug_factor > 1:
        x_train, y_train_dict = augment_training_data(
            x_train,
            y_train_dict,
            aug_factor=args.aug_factor,
            noise_level=args.noise_level,
            seed=args.seed,
        )
    y_train_aug = np.column_stack([y_train_dict[name] for name in TARGET_KEYS])
    target_standard_deviations = _compute_target_standard_deviations(y_train_dict, TARGET_KEYS)
    loss_weights = compute_inverse_variance_loss_weights(y_train_dict, target_keys=TARGET_KEYS)
    scalers = _target_scalers_from_npz(data)

    rows: list[dict[str, float | int | str]] = []
    for variant in variants:
        _set_seed(args.seed)
        print(f"\n{'=' * 72}\nRunning paper-model ablation: {variant}\n{'=' * 72}")
        if variant == "no_multitask":
            predictions, params = _train_no_multitask_variant(
                x_train=x_train,
                y_train=y_train_aug,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                args=args,
            )
        else:
            predictions, params = _train_multitask_variant(
                variant=variant,
                use_residual=False,
                use_layernorm=variant != "no_residual_no_layernorm",
                x_train=x_train,
                y_train=y_train_dict,
                x_val=x_val,
                y_val=y_val_dict,
                x_test=x_test,
                target_standard_deviations=target_standard_deviations,
                loss_weights=loss_weights,
                args=args,
            )
        mae = _physical_mae(predictions, y_test, scalers)
        rows.append({"variant": variant, "params": params, **{f"MAE_{key}": value for key, value in mae.items()}})

    summary = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "variants": variants,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "augmentation": {"aug_factor": args.aug_factor, "noise_level": args.noise_level},
                "paper_hyperparameters": FixedPaperHP.values,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved ablation summary to: {summary_path}")
    print(summary.to_string(index=False))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ablations for this repository's paper model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--aug-factor", type=int, default=3)
    parser.add_argument("--noise-level", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--use-xla", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    return parser


def main() -> None:
    run_paper_ablations(build_argparser().parse_args())


if __name__ == "__main__":
    main()
