"""Bootstrap test MAEs of the already-trained paper model, without fitting it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "paper_hyperparams_model.keras"
DEFAULT_DATA = PROJECT_ROOT / "data" / "sdss_dr12_processed_flux_benchmark.npz"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "data" / "paper_hyperparams_test_predictions.npz"
DEFAULT_RESULTS = PROJECT_ROOT / "results_and_evaluations" / "paper_hyperparams_bootstrap_mae.json"
TARGET_NAMES = ("Teff", "[Fe/H]", "log g")
OUTPUT_NAMES = ("teff_output", "feh_output", "logg_output")
UNITS = ("K", "dex", "dex")


def bootstrap_mae(abs_errors: np.ndarray, n_boot: int = 10_000, seed: int = 42) -> tuple[float, float, float]:
    """Percentile interval from sample-sized resamples of absolute errors."""
    errors = np.asarray(abs_errors, dtype=np.float64).reshape(-1)
    if errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError("Absolute errors must be nonempty and finite.")
    if n_boot < 1:
        raise ValueError("n_boot must be positive.")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    n = errors.size
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = errors[idx].mean()
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(errors.mean()), float(low), float(high)


def _prediction_matrix(model: tf.keras.Model, features: np.ndarray, batch_size: int) -> np.ndarray:
    raw = model.predict(features, batch_size=batch_size, verbose=0)
    if isinstance(raw, dict):
        columns = [np.asarray(raw[name]).reshape(-1) for name in OUTPUT_NAMES]
    elif isinstance(raw, (list, tuple)):
        by_name = dict(zip(model.output_names, raw))
        columns = [np.asarray(by_name[name]).reshape(-1) for name in OUTPUT_NAMES]
    else:
        matrix = np.asarray(raw)
        if matrix.shape != (len(features), len(TARGET_NAMES)):
            raise ValueError(f"Unexpected model prediction shape: {matrix.shape}")
        return matrix.astype(np.float64)
    return np.column_stack(columns).astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--predictions-out", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--results-out", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    with np.load(args.data, allow_pickle=False) as data:
        x_test = np.asarray(data["X_test_features"], dtype=np.float32)
        y_scaled = np.asarray(data["y_test_targets"], dtype=np.float64)
        center = np.asarray(data["label_robust_center"], dtype=np.float64)
        scale = np.asarray(data["label_robust_scale"], dtype=np.float64)
        test_ids = np.asarray(data["test_ids"])
    n = len(x_test)
    if y_scaled.shape != (n, 3) or center.shape != (3,) or scale.shape != (3,) or test_ids.shape != (n,):
        raise ValueError("Test features, targets, IDs, or scaler parameters have unexpected dimensions.")
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid RobustScaler parameters in the dataset.")

    model = tf.keras.models.load_model(args.model, compile=False)
    pred_scaled = _prediction_matrix(model, x_test, args.batch_size)
    if pred_scaled.shape != (n, 3) or not np.isfinite(pred_scaled).all():
        raise ValueError("Model predictions have unexpected dimensions or nonfinite values.")
    y_true = y_scaled * scale + center
    y_pred = pred_scaled * scale + center
    if not np.isfinite(y_true).all():
        raise ValueError("Physical test targets contain nonfinite values.")

    # Rows and test IDs stay aligned, so another model can use paired resamples.
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.predictions_out,
        test_ids=test_ids,
        target_names=np.asarray(TARGET_NAMES),
        y_true=y_true,
        y_pred=y_pred,
    )

    metrics = {}
    print(f"Model: {args.model.resolve()}")
    print(f"Test samples: {n}; bootstrap resamples: {args.n_boot}; seed: {args.seed}")
    for col, (name, unit) in enumerate(zip(TARGET_NAMES, UNITS)):
        abs_errors = np.abs(y_true[:, col] - y_pred[:, col])
        mae, low, high = bootstrap_mae(abs_errors, args.n_boot, args.seed)
        metrics[name] = {"mae": mae, "ci_95": [low, high], "unit": unit}
        print(f"{name}: MAE = {mae:.6f} {unit}; IC 95% = [{low:.6f}, {high:.6f}] {unit}")

    result = {
        "model_path": str(args.model.resolve()),
        "dataset_path": str(args.data.resolve()),
        "predictions_path": str(args.predictions_out.resolve()),
        "n_test": n,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "method": "percentile bootstrap of test-set absolute errors",
        "metrics": metrics,
    }
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Predictions: {args.predictions_out.resolve()}")
    print(f"Results: {args.results_out.resolve()}")


if __name__ == "__main__":
    main()
