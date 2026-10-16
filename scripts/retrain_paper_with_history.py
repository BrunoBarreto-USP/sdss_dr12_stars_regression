"""Train the recorded best-trial architecture and retain one consistent experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.train_paper_hyperparams import FixedPaperHP, build_paper_fixed_model
from training.train_best_tuner_trial import _load_best_trial
from training.training import (
    NormalizedMaeSumCallback,
    _create_dataset,
    compute_inverse_variance_loss_weights,
)
from data_preprocessing.augmentation import (
    augment_training_data,
    prepare_multitask_dicts,
)
from data_preprocessing.validation import validate_compact_dataset

KEYS = ("teff_output", "feh_output", "logg_output")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/paper_figures/single_column/corrected_best_trial",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1:
        parser.error("Epochs, batch size and thread count must be positive.")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "history.csv").exists():
        raise FileExistsError(f"Existing training history must be preserved: {out}")
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.mixed_precision.set_global_policy("float32")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    source = ROOT / "data/sdss_dr12_processed_flux_benchmark.npz"
    z = np.load(source)
    validate_compact_dataset(z, ROOT / "data")
    x = z["X_train_features"]
    y = z["y_train_targets"]
    xv, yv = z["X_val_features"], z["y_val_targets"]
    trial_folder = (
        ROOT / "results_and_evaluations/keras_tuner/stellar_baseline_orchestration"
    )
    trial_id, trial_score, hp = _load_best_trial(trial_folder)
    if FixedPaperHP.values != hp:
        raise ValueError("Paper hyperparameters differ from the recorded best trial.")
    print(f"Preparing corrected final training of {trial_id}.", flush=True)
    yt, yvt = prepare_multitask_dicts(y, yv)
    xa, yt = augment_training_data(
        x, yt, aug_factor=3, noise_level=0.05, seed=args.seed
    )
    std = {k: float(np.std(yt[k], dtype=np.float64)) for k in KEYS}
    weights = compute_inverse_variance_loss_weights(yt, KEYS)
    steps = len(xa) // args.batch_size
    if steps < 1:
        raise ValueError("Batch size exceeds the number of training examples.")
    model = build_paper_fixed_model(
        input_dim=4000,
        total_steps=steps * args.epochs,
        loss_weights=weights,
        batch_size=args.batch_size,
        jit_compile=False,
    )
    count = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    heads = {
        label: [
            layer.units
            for layer in model.layers
            if layer.name.startswith(key + "_head_dense_")
        ]
        for label, key in zip(("Teff", "[Fe/H]", "log g"), KEYS)
    }
    assert heads == {"Teff": [16], "[Fe/H]": [48, 32, 16], "log g": [48, 32, 16]}, heads
    assert count == 530275, count
    config = {
        "purpose": "Corrected final training of the Bayesian-selected architecture; curves, weights, predictions and intervals refer to its validation-selected checkpoint.",
        "architecture_source": str(trial_folder / trial_id / "trial.json"),
        "source_trial_id": trial_id,
        "source_trial_score": trial_score,
        "hyperparameters": hp,
        "trainable_parameters": count,
        "epochs": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "steps_per_epoch": steps,
        "original_train_samples": len(x),
        "augmented_train_samples": len(xa),
        "validation_samples": len(xv),
        "loss_weights": weights,
        "loss_weighting": "inverse variance of robust-scaled training targets, normalized to sum 3",
        "selection": "sum of scaled validation MAEs divided by scaled training target standard deviations",
        "target_standard_deviations": std,
        "heads": heads,
        "learning_rate": 0.005,
        "weight_decay": 1e-5,
        "augmentation": {"factor": 3, "noise_fraction": [0.01, 0.05]},
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    train = _create_dataset(
        xa, yt, batch_size=args.batch_size, shuffle=True, cache=False
    )
    validation = _create_dataset(
        xv, yvt, batch_size=args.batch_size, shuffle=False, cache=False
    )
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    train = train.with_options(options)
    validation = validation.with_options(options)
    monitor = "val_normalized_mae_sum"
    callbacks = [
        NormalizedMaeSumCallback(std, KEYS),
        tf.keras.callbacks.ModelCheckpoint(
            str(out / "best.weights.h5"),
            monitor=monitor,
            mode="min",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.CSVLogger(str(out / "history.csv")),
    ]
    print(
        f"Training {args.epochs} epochs; {steps} steps/epoch; inverse-variance loss weights {weights}",
        flush=True,
    )
    history = model.fit(
        train,
        validation_data=validation,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    raw = {k: [float(v) for v in values] for k, values in history.history.items()}
    (out / "history.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    best = int(np.nanargmin(raw[monitor]))
    model.load_weights(out / "best.weights.h5")
    export = tf.keras.Model(inputs=model.inputs, outputs=model.outputs, name=model.name)
    export.save(out / "retrained_paper_model.keras")
    pred = model.predict(z["X_test_features"], batch_size=args.batch_size, verbose=0)
    pred = np.column_stack([np.asarray(p).ravel() for p in pred]).astype(np.float64)
    pred = pred * z["label_robust_scale"] + z["label_robust_center"]
    # Use exactly the same physical reference labels as the archived baselines.
    reference = np.load(ROOT / "data/paper_hyperparams_test_predictions.npz")
    np.testing.assert_array_equal(reference["test_ids"], z["test_ids"])
    true = reference["y_true"].astype(float)
    np.testing.assert_allclose(
        true, z["y_test_targets_original"], atol=0.001, rtol=1e-6
    )
    mae = np.mean(abs(pred - true), axis=0)
    np.savez_compressed(
        out / "test_predictions.npz",
        test_ids=z["test_ids"],
        target_names=np.asarray(("Teff", "[Fe/H]", "log g")),
        y_true=true,
        y_pred=pred,
    )
    config.update(
        {
            "selected_epoch": best + 1,
            "best_validation_score": raw[monitor][best],
            "test_mae": dict(zip(("Teff", "[Fe/H]", "log g"), mae.tolist())),
        }
    )
    (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Finished:", json.dumps(config), flush=True)


if __name__ == "__main__":
    main()
