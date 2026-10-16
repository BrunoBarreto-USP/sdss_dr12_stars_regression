"""Compare saved CNN and Li-DNN checkpoints with the paper model on paired test rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sdss_dr12_processed_flux_benchmark.npz"
PAPER_PREDICTIONS = ROOT / "data" / "paper_hyperparams_test_predictions.npz"
CNN_MODEL = ROOT / "models" / "benchmarks" / "fabbro_cnn.keras"
DNN_MODELS = [ROOT / "models" / "benchmarks" / f"li_dnn_{name}.keras" for name in ("teff", "feh", "logg")]
RESULTS = ROOT / "results_and_evaluations" / "paired_bootstrap_baselines.json"
NAMES = ("Teff", "[Fe/H]", "log g")
UNITS = ("K", "dex", "dex")


def _predict(path: Path, features: np.ndarray, batch_size: int) -> np.ndarray:
    model = tf.keras.models.load_model(path, compile=False)
    result = np.asarray(model.predict(features, batch_size=batch_size, verbose=0), dtype=np.float64)
    del model
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.n_boot < 1 or args.batch_size < 1:
        parser.error("--n-boot and --batch-size must be positive")

    with np.load(DATA, allow_pickle=False) as data:
        features = np.asarray(data["X_test_features"], dtype=np.float32)
        ids = np.asarray(data["test_ids"])
        y_scaled = np.asarray(data["y_test_targets"], dtype=np.float64)
        centers = np.asarray(data["label_robust_center"], dtype=np.float64)
        scales = np.asarray(data["label_robust_scale"], dtype=np.float64)
    with np.load(PAPER_PREDICTIONS, allow_pickle=False) as paper:
        paper_ids = np.asarray(paper["test_ids"])
        y_true = np.asarray(paper["y_true"], dtype=np.float64)
        paper_pred = np.asarray(paper["y_pred"], dtype=np.float64)
    n = len(ids)
    if not np.array_equal(ids, paper_ids) or y_true.shape != (n, 3) or paper_pred.shape != (n, 3):
        raise ValueError("Paper predictions do not align with the current test set.")
    if not np.allclose(y_true, y_scaled * scales + centers, rtol=0, atol=1e-8):
        raise ValueError("Paper targets differ from the current test-set targets.")

    cnn_scaled = _predict(CNN_MODEL, features[..., np.newaxis], args.batch_size)
    dnn_scaled = np.column_stack([
        _predict(path, features, args.batch_size).reshape(-1) for path in DNN_MODELS
    ])
    if cnn_scaled.shape != (n, 3) or dnn_scaled.shape != (n, 3):
        raise ValueError("Unexpected CNN or DNN prediction shape.")
    if not np.isfinite(cnn_scaled).all() or not np.isfinite(dnn_scaled).all():
        raise ValueError("CNN or DNN predictions contain nonfinite values.")

    predictions = {
        "CNN": cnn_scaled * scales + centers,
        "DNN": dnn_scaled * scales + centers,
    }
    for name, pred in predictions.items():
        path = ROOT / "data" / f"{name.lower()}_test_predictions.npz"
        np.savez_compressed(path, test_ids=ids, target_names=np.asarray(NAMES), y_true=y_true, y_pred=pred)
        print(f"Saved {name} predictions: {path}")

    paper_errors = np.abs(y_true - paper_pred)
    errors = {name: np.abs(y_true - pred) for name, pred in predictions.items()}
    # Positive difference means the paper model has a lower MAE.
    differences = np.stack([errors[name] - paper_errors for name in predictions], axis=2)
    rng = np.random.default_rng(args.seed)
    boot = np.empty((args.n_boot, 3, len(predictions)), dtype=np.float64)
    for b in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = differences[idx].mean(axis=0)
    ci = np.percentile(boot, [2.5, 97.5], axis=0)

    report = {"n_test": n, "n_boot": args.n_boot, "seed": args.seed,
              "difference_definition": "MAE_baseline minus MAE_paper; positive favors paper model",
              "comparisons": {}}
    for model_idx, name in enumerate(predictions):
        report["comparisons"][name] = {}
        for target_idx, (target, unit) in enumerate(zip(NAMES, UNITS)):
            delta = float(differences[:, target_idx, model_idx].mean())
            low, high = (float(ci[0, target_idx, model_idx]), float(ci[1, target_idx, model_idx]))
            report["comparisons"][name][target] = {
                "mae_paper": float(paper_errors[:, target_idx].mean()),
                "mae_baseline": float(errors[name][:, target_idx].mean()),
                "mae_difference": delta,
                "difference_ci_95": [low, high],
                "ci_excludes_zero": bool(low > 0 or high < 0),
                "unit": unit,
            }
            print(f"{name} - {target}: difference = {delta:.6f} {unit}; IC 95% = [{low:.6f}, {high:.6f}] {unit}")
    RESULTS.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
