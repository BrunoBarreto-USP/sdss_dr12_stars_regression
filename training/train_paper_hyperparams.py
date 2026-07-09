"""Train the paper-selected multitask model without Bayesian search."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

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
    build_export_model,
)
from training.training import (
    NORMALIZED_MAE_SUM_NAME,
    NormalizedMaeSumCallback,
    _compute_target_variances,
    _create_dataset,
    _score_from_eval_results,
    cast_target_dicts_to_float32,
    compute_inverse_variance_loss_weights,
    configure_cpu_runtime,
    configure_gpu_runtime,
)


DEFAULT_DATA = PROJECT_ROOT / "data" / "sdss_dr12_processed_flux_benchmark.npz"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "paper_hyperparams_model.keras"
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "paper_hyperparams_best.weights.h5"
DEFAULT_RESULTS = PROJECT_ROOT / "results_and_evaluations" / "paper_hyperparams_metrics.json"
TARGET_KEYS = ("teff_output", "feh_output", "logg_output")
TARGET_LABELS = ("Teff", "[Fe/H]", "log g")


class FixedPaperHP:
    """Paper-selected hyperparameters."""

    values = {
        "initial_units": 128,
        "initial_dropout": 0.0,
        "shared_blocks": 1,
        "block_units": 64,
        "trunk_dropout": 0.4,
        "teff_output_head_template": "(48, 32)",
        "teff_output_head_dropout": 0.0,
        "feh_output_head_template": "(16,)",
        "feh_output_head_dropout": 0.0,
        "logg_output_head_template": "(48,)",
        "logg_output_head_dropout": 0.0,
        "learning_rate": 7.0e-4,
        "weight_decay": 1.0e-2,
    }


def _target_scalers_from_npz(data) -> list:
    from sklearn.preprocessing import RobustScaler

    scalers = []
    for center, scale in zip(data["label_robust_center"], data["label_robust_scale"]):
        scaler = RobustScaler()
        scaler.center_ = np.asarray([center], dtype=np.float64)
        scaler.scale_ = np.asarray([scale], dtype=np.float64)
        scaler.n_features_in_ = 1
        scalers.append(scaler)
    return scalers


def _stack_predictions(predictions) -> np.ndarray:
    if isinstance(predictions, list):
        return np.column_stack([np.asarray(pred).reshape(-1) for pred in predictions])
    return np.asarray(predictions)


def _inverse_targets(y_scaled: np.ndarray, scalers: list) -> np.ndarray:
    y = np.empty_like(y_scaled, dtype=np.float64)
    for idx, scaler in enumerate(scalers):
        y[:, idx] = scaler.inverse_transform(y_scaled[:, idx].reshape(-1, 1)).ravel()
    return y


def _physical_mae(model, x: np.ndarray, y_scaled: np.ndarray, scalers: list, *, batch_size: int) -> dict[str, float]:
    pred_scaled = _stack_predictions(model.predict(x, batch_size=batch_size, verbose=0))
    pred = _inverse_targets(pred_scaled, scalers)
    true = _inverse_targets(y_scaled, scalers)
    return {
        label: float(np.mean(np.abs(pred[:, idx] - true[:, idx])))
        for idx, label in enumerate(TARGET_LABELS)
    }


def _history_min(history: tf.keras.callbacks.History, key: str) -> float | None:
    values = history.history.get(key)
    if not values:
        return None
    return float(np.nanmin(values))


def _dense_head(x, *, output_name: str, units: tuple[int, ...]):
    for idx, width in enumerate(units):
        x = tf.keras.layers.Dense(
            width,
            activation="gelu",
            name=f"{output_name}_head_dense_{idx}",
        )(x)
    return tf.keras.layers.Dense(1, activation="linear", name=output_name, dtype="float32")(x)


def build_paper_fixed_model(
    *,
    input_dim: int,
    total_steps: int,
    loss_weights: dict[str, float],
    batch_size: int,
    jit_compile: bool,
) -> tf.keras.Model:
    """Build the 542,771-parameter paper/no-gating architecture."""
    hp = FixedPaperHP.values
    inputs = tf.keras.layers.Input(shape=(input_dim,), name="stellar_input")

    x = tf.keras.layers.Dense(hp["initial_units"], name="input_stem_dense")(inputs)
    x = tf.keras.layers.LayerNormalization(name="input_stem_ln")(x)
    x = tf.keras.layers.Activation("gelu", name="input_stem_activation")(x)

    shortcut = x
    x = tf.keras.layers.LayerNormalization(name="shared_block_0_ln_0")(x)
    x = tf.keras.layers.Dense(hp["block_units"], name="shared_block_0_dense_0")(x)
    x = tf.keras.layers.Activation("gelu", name="shared_block_0_activation_0")(x)
    x = tf.keras.layers.LayerNormalization(name="shared_block_0_ln_1")(x)
    x = tf.keras.layers.Dense(hp["block_units"], name="shared_block_0_dense_1")(x)
    x = tf.keras.layers.Dropout(hp["trunk_dropout"], name="shared_block_0_dropout")(x)

    shortcut = tf.keras.layers.LayerNormalization(name="shared_block_0_shortcut_ln")(shortcut)
    shortcut = tf.keras.layers.Dense(hp["block_units"], name="shared_block_0_shortcut_dense")(shortcut)
    x = tf.keras.layers.Add(name="shared_block_0_add")([x, shortcut])
    shared = tf.keras.layers.LayerNormalization(name="trunk_final_ln")(x)

    outputs = [
        _dense_head(shared, output_name="teff_output", units=(48, 32)),
        _dense_head(shared, output_name="feh_output", units=(16,)),
        _dense_head(shared, output_name="logg_output", units=(48,)),
    ]
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="stellar_model")

    warmup_steps = int(total_steps * WARMUP_FRACTION)
    learning_rate = WarmupCosineDecay(
        hp["learning_rate"] * (batch_size / 256) ** 0.5,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        alpha=DEFAULT_COSINE_ALPHA,
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=hp["weight_decay"],
            global_clipnorm=5.0,
        ),
        loss={name: tf.keras.losses.Huber(delta=DEFAULT_HUBER_DELTA) for name in TARGET_KEYS},
        loss_weights=loss_weights,
        metrics={name: [tf.keras.metrics.MeanAbsoluteError(name="mae")] for name in TARGET_KEYS},
        jit_compile=jit_compile,
    )
    return model


def train_paper_model(args: argparse.Namespace) -> dict[str, object]:
    if args.cpu:
        runtime = configure_cpu_runtime()
    else:
        runtime = configure_gpu_runtime(require_gpu=False, enable_mixed_precision=not args.no_mixed_precision)
    print("Runtime:", runtime)

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

    target_variances = _compute_target_variances(y_train_dict, TARGET_KEYS)
    loss_weights = compute_inverse_variance_loss_weights(y_train_dict, target_keys=TARGET_KEYS)
    total_steps = int(x_train.shape[0] // args.batch_size) * args.epochs

    print("Paper fixed hyperparameters:")
    for key, value in FixedPaperHP.values.items():
        print(f"  {key}: {value}")
    print(f"epochs: {args.epochs}, batch_size: {args.batch_size}, total_steps: {total_steps}")
    print("loss_weights:", loss_weights)

    train_dataset = _create_dataset(x_train, y_train_dict, batch_size=args.batch_size, shuffle=True, cache=args.cache)
    val_dataset = _create_dataset(x_val, y_val_dict, batch_size=args.batch_size, shuffle=False, cache=args.cache)
    model = build_paper_fixed_model(
        input_dim=x_train.shape[1],
        loss_weights=loss_weights,
        total_steps=total_steps,
        jit_compile=args.use_xla,
        batch_size=args.batch_size,
    )
    model.summary()
    print(f"Trainable parameters: {model.count_params():,}")

    args.weights.parent.mkdir(parents=True, exist_ok=True)
    monitor = f"val_{NORMALIZED_MAE_SUM_NAME}"
    callbacks = [
        NormalizedMaeSumCallback(target_variances, TARGET_KEYS),
        tf.keras.callbacks.ModelCheckpoint(
            str(args.weights),
            monitor=monitor,
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    if args.weights.exists():
        model.load_weights(args.weights)

    val_eval = model.evaluate(val_dataset, return_dict=True, verbose=0)
    val_normalized_mae_sum = _score_from_eval_results(
        val_eval,
        target_variances=target_variances,
        target_keys=TARGET_KEYS,
    )
    scalers = _target_scalers_from_npz(data)
    val_mae = _physical_mae(model, x_val, y_val, scalers, batch_size=args.batch_size)
    test_mae = _physical_mae(model, x_test, y_test, scalers, batch_size=args.batch_size)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    export_model = build_export_model(model)
    export_model.save(args.model)

    metrics = {
        "model_path": str(args.model),
        "weights_path": str(args.weights),
        "trainable_parameters": int(model.count_params()),
        "best_history_val_normalized_mae_sum": _history_min(history, monitor),
        "reloaded_best_val_normalized_mae_sum": float(val_normalized_mae_sum),
        "validation_mae": val_mae,
        "test_mae": test_mae,
        "hyperparameters": dict(FixedPaperHP.values),
        "augmentation": {"aug_factor": args.aug_factor, "noise_level": args.noise_level},
    }

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nPaper fixed-hyperparameter results")
    print(f"  best history {monitor}: {metrics['best_history_val_normalized_mae_sum']:.6f}")
    print(f"  reloaded best {monitor}: {metrics['reloaded_best_val_normalized_mae_sum']:.6f}")
    print("  validation MAE:")
    for label, value in val_mae.items():
        print(f"    {label}: {value:.6f}")
    print("  test MAE:")
    for label, value in test_mae.items():
        print(f"    {label}: {value:.6f}")
    print(f"  metrics JSON: {args.results}")
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the paper-selected fixed hyperparameter model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--aug-factor", type=int, default=3)
    parser.add_argument("--noise-level", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only training.")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision on GPU.")
    parser.add_argument("--use-xla", action="store_true")
    parser.add_argument("--cache", action="store_true", help="Cache tf.data datasets in memory.")
    return parser


def main() -> None:
    train_paper_model(build_argparser().parse_args())


if __name__ == "__main__":
    main()
