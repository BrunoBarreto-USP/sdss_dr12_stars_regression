"""Retrain the best recorded Keras Tuner trial for a fixed epoch budget."""

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
from model_definitions.model_definitions import DEFAULT_COSINE_ALPHA, build_export_model, build_model
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
DEFAULT_TUNER_PROJECT = PROJECT_ROOT / "results_and_evaluations" / "keras_tuner" / "stellar_baseline_orchestration"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "best_tuner_trial_100epochs.keras"
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "best_tuner_trial_100epochs.weights.h5"
DEFAULT_RESULTS = PROJECT_ROOT / "results_and_evaluations" / "best_tuner_trial_100epochs_metrics.json"
TARGET_KEYS = ("teff_output", "feh_output", "logg_output")
TARGET_LABELS = ("Teff", "FeH", "logg")


class FixedTrialHP:
    """Keras-Tuner-compatible wrapper around saved trial hyperparameter values."""

    def __init__(self, values: dict[str, object]):
        self.values = dict(values)

    def Choice(self, name, values, default=None, **_kwargs):
        value = self.values[name]
        if value not in values:
            raise ValueError(f"Saved value {value!r} for {name} is not in {values!r}")
        return value

    def Float(self, name, min_value, max_value, default=None, **_kwargs):
        value = float(self.values[name])
        if not (float(min_value) <= value <= float(max_value)):
            raise ValueError(f"Saved value {value!r} for {name} is outside [{min_value}, {max_value}]")
        return value

    def Int(self, name, min_value, max_value, default=None, **_kwargs):
        value = int(self.values[name])
        if not (int(min_value) <= value <= int(max_value)):
            raise ValueError(f"Saved value {value!r} for {name} is outside [{min_value}, {max_value}]")
        return value


def _load_best_trial(tuner_project_dir: Path) -> tuple[str, float, dict[str, object]]:
    best: tuple[str, float, dict[str, object]] | None = None
    for trial_path in sorted(tuner_project_dir.glob("trial_*/trial.json")):
        with trial_path.open("r", encoding="utf-8") as fh:
            trial = json.load(fh)
        if trial.get("status") != "COMPLETED":
            continue
        score = trial.get("score")
        values = trial.get("hyperparameters", {}).get("values")
        if score is None or values is None:
            continue
        item = (trial_path.parent.name, float(score), values)
        if best is None or item[1] < best[1]:
            best = item
    if best is None:
        raise RuntimeError(f"No completed trials with scores found under {tuner_project_dir}")
    return best


def _inverse(values: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return values.astype(np.float64) * scales.reshape(1, -1) + centers.reshape(1, -1)


def _stack_predictions(predictions) -> np.ndarray:
    if isinstance(predictions, (list, tuple)):
        return np.column_stack([np.asarray(pred).reshape(-1) for pred in predictions])
    return np.asarray(predictions)


def _physical_mae(model: tf.keras.Model, x: np.ndarray, y_scaled: np.ndarray, data, *, batch_size: int) -> dict[str, float]:
    pred_scaled = _stack_predictions(model.predict(x, batch_size=batch_size, verbose=0))
    centers = np.asarray(data["label_robust_center"], dtype=np.float64)
    scales = np.asarray(data["label_robust_scale"], dtype=np.float64)
    pred = _inverse(pred_scaled, centers, scales)
    if "y_test_targets_original" in data.files and y_scaled.shape[0] == data["y_test_targets_original"].shape[0]:
        true = np.asarray(data["y_test_targets_original"], dtype=np.float64)
    else:
        true = _inverse(y_scaled, centers, scales)
    return {
        name: float(np.mean(np.abs(pred[:, idx] - true[:, idx])))
        for idx, name in enumerate(TARGET_LABELS)
    }


def train_best_trial(args: argparse.Namespace) -> dict[str, object]:
    if args.cpu:
        runtime = configure_cpu_runtime()
    else:
        runtime = configure_gpu_runtime(require_gpu=False, enable_mixed_precision=not args.no_mixed_precision)
    print("Runtime:", runtime)

    trial_id, trial_score, hp_values = _load_best_trial(args.tuner_project)
    print(f"Best trial: {trial_id}, recorded score={trial_score:.6f}")
    print("Hyperparameters:")
    for key in sorted(hp_values):
        print(f"  {key}: {hp_values[key]}")

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

    model = build_model(
        input_shape=x_train.shape[1],
        num_outputs=len(TARGET_KEYS),
        hp=FixedTrialHP(hp_values),
        loss_weights=loss_weights,
        total_steps=total_steps,
        cosine_alpha=DEFAULT_COSINE_ALPHA,
        use_lr_schedule=True,
        jit_compile=args.use_xla,
        batch_size=args.batch_size,
    )
    print(f"Trainable parameters: {model.count_params():,}")

    train_dataset = _create_dataset(x_train, y_train_dict, batch_size=args.batch_size, shuffle=True, cache=args.cache)
    val_dataset = _create_dataset(x_val, y_val_dict, batch_size=args.batch_size, shuffle=False, cache=args.cache)
    monitor = f"val_{NORMALIZED_MAE_SUM_NAME}"
    args.weights.parent.mkdir(parents=True, exist_ok=True)
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
    history = model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=callbacks, verbose=1)
    if args.weights.exists():
        model.load_weights(args.weights)

    val_eval = model.evaluate(val_dataset, return_dict=True, verbose=0)
    val_normalized_mae_sum = _score_from_eval_results(
        val_eval,
        target_variances=target_variances,
        target_keys=TARGET_KEYS,
    )
    val_mae = _physical_mae(model, x_val, y_val, data, batch_size=args.batch_size)
    test_mae = _physical_mae(model, x_test, y_test, data, batch_size=args.batch_size)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    build_export_model(model).save(args.model)

    results = {
        "best_trial": trial_id,
        "recorded_trial_score": trial_score,
        "hyperparameters": hp_values,
        "model_path": str(args.model),
        "weights_path": str(args.weights),
        "epochs": args.epochs,
        "augmentation": {"aug_factor": args.aug_factor, "noise_level": args.noise_level},
        "trainable_parameters": int(model.count_params()),
        "best_history_val_normalized_mae_sum": float(np.nanmin(history.history[monitor])),
        "reloaded_best_val_normalized_mae_sum": float(val_normalized_mae_sum),
        "validation_mae": val_mae,
        "test_mae": test_mae,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Saved model: {args.model}")
    print(f"Saved metrics: {args.results}")
    print(f"Reloaded best {monitor}: {val_normalized_mae_sum:.6f}")
    print("Test MAE:", test_mae)
    return results


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrain the best saved Keras Tuner trial.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tuner-project", type=Path, default=DEFAULT_TUNER_PROJECT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--aug-factor", type=int, default=3)
    parser.add_argument("--noise-level", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--use-xla", action="store_true")
    parser.add_argument("--cache", action="store_true")
    return parser


def main() -> None:
    train_best_trial(build_argparser().parse_args())


if __name__ == "__main__":
    main()
