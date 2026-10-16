"""Audit the selected checkpoint and reuse the project's existing bootstrap methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from results_and_evaluations.bootstrap_paper_model_mae import bootstrap_mae
from results_and_evaluations import paired_bootstrap_all_models
from results_and_evaluations.measure_madds import estimate_model_flops
from training.train_best_tuner_trial import _load_best_trial
from data_preprocessing.validation import validate_compact_dataset

TARGETS = ("Teff", "[Fe/H]", "log g")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "outputs/paper_figures/single_column/corrected_best_trial",
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    run = args.run_dir
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    trial, score, hp = _load_best_trial(
        ROOT / "results_and_evaluations/keras_tuner/stellar_baseline_orchestration"
    )
    assert hp == config["hyperparameters"] and trial == config["source_trial_id"]
    history = pd.read_csv(run / "history.csv")
    selected = int(history.val_normalized_mae_sum.argmin()) + 1
    assert selected == config["selected_epoch"]
    assert len(history) == config["epochs"] and history.epoch.tolist() == list(
        range(config["epochs"])
    )
    path = run / "retrained_paper_model.keras"
    model = tf.keras.models.load_model(path, compile=False)
    actual = {
        target: [
            layer.units
            for layer in model.layers
            if layer.name.startswith(key + "_head_dense_")
        ]
        for target, key in zip(TARGETS, ("teff_output", "feh_output", "logg_output"))
    }
    assert (
        actual
        == config["heads"]
        == {"Teff": [16], "[Fe/H]": [48, 32, 16], "log g": [48, 32, 16]}
    )
    assert model.count_params() == 530275
    z = np.load(run / "test_predictions.npz")
    original = np.load(ROOT / "data/paper_hyperparams_test_predictions.npz")
    np.testing.assert_array_equal(z["test_ids"], original["test_ids"])
    np.testing.assert_array_equal(z["y_true"], original["y_true"])
    assert len(np.unique(z["test_ids"])) == 14443
    data = np.load(ROOT / "data/sdss_dr12_processed_flux_benchmark.npz")
    validate_compact_dataset(data, ROOT / "data")
    small = np.column_stack(
        [
            np.asarray(p).ravel()
            for p in model.predict(data["X_test_features"][:64], verbose=0)
        ]
    ).astype(np.float64)
    small = small * data["label_robust_scale"] + data["label_robust_center"]
    np.testing.assert_allclose(small, z["y_pred"][:64], atol=0.002, rtol=1e-6)
    errors = abs(z["y_pred"] - z["y_true"])
    metrics = {}
    for j, (target, unit) in enumerate(zip(TARGETS, ("K", "dex", "dex"))):
        mean, low, high = bootstrap_mae(errors[:, j], args.n_boot, 42)
        np.testing.assert_allclose(mean, config["test_mae"][target], rtol=1e-12)
        metrics[target] = {"mae": mean, "ci_95": [low, high], "unit": unit}
    report = {
        "model_path": str(path),
        "predictions_path": str(run / "test_predictions.npz"),
        "model_sha256": sha256(path),
        "n_test": len(errors),
        "n_boot": args.n_boot,
        "seed": 42,
        "method": "percentile bootstrap of test-set absolute errors",
        "metrics": metrics,
    }
    (run / "bootstrap_mae.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    saved_argv = sys.argv
    try:
        sys.argv = [
            "paired_bootstrap_all_models",
            "--n-boot",
            str(args.n_boot),
            "--seed",
            "42",
            "--paper-predictions",
            str(run / "test_predictions.npz"),
            "--results-out",
            str(run / "paired_bootstrap.json"),
            "--no-message",
        ]
        paired_bootstrap_all_models.main()
    finally:
        sys.argv = saved_argv
    params, flops, macs = estimate_model_flops(path, 4000)
    pd.DataFrame(
        [
            {
                "model": str(path),
                "input_dim": 4000,
                "params": params,
                "flops": flops,
                "macs": macs,
            }
        ]
    ).to_csv(run / "complexity.csv", index=False)
    audit = {
        "source_trial": trial,
        "source_trial_score": score,
        "hyperparameters": hp,
        "actual_heads": actual,
        "trainable_parameters": params,
        "flops": flops,
        "model_path": str(path),
        "model_sha256": sha256(path),
        "predictions_path": str(run / "test_predictions.npz"),
        "predictions_sha256": sha256(run / "test_predictions.npz"),
        "epochs": len(history),
        "selected_epoch": selected,
        "n_test": len(errors),
        "validation_selected_checkpoint": True,
        "same_checkpoint_for_curves_weights_predictions": True,
        "previous_paper_model_sha256": sha256(
            ROOT / "models/paper_hyperparams_model.keras"
        ),
        "previous_paired_bootstrap_preserved": str(
            ROOT / "results_and_evaluations/paired_bootstrap_all_models.json"
        ),
    }
    (run / "architecture_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(
        "Corrected experiment finalized:",
        json.dumps(
            {
                "parameters": params,
                "flops": flops,
                "selected_epoch": selected,
                "metrics": metrics,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
