"""Shared fitted-model lifecycle; numerical builders remain in the domain packages."""
import ast
import gc
from importlib import metadata
import json
import os
from pathlib import Path
import random
import subprocess

import numpy as np

from .config import OUTPUT_NAMES, ROOT, TARGET_NAMES, load_config
from .predictions import physical_labels, write_predictions
from .runs import Run, sha256, write_json


def runtime(config):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(config["threads"])
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass  # Existing notebook/test runtime keeps its already-initialized pool.
    tf.keras.mixed_precision.set_global_policy("float32")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    tf.keras.utils.set_random_seed(config["seed"])
    tf.config.experimental.enable_op_determinism()
    return tf


def software_versions():
    import platform
    result = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "scikit-learn", "pyarrow",
                 "tensorflow", "keras", "keras-tuner", "huggingface_hub", "datasets", "torch"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return result


def source_provenance():
    import hashlib
    folders = ("stellar_workflow", "training", "model_definitions", "data_acquisition",
               "data_preprocessing", "data_exploration", "results_and_evaluations", "finetuning", "scripts")
    sources = {p.relative_to(ROOT).as_posix(): sha256(p) for d in folders for p in sorted((ROOT / d).glob("*.py"))}
    result = {"source_sha256": sources}
    try:
        result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        result["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
        result["git_dirty"] = None
    result["source_tree_sha256"] = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return result


def _heads(hp):
    return {target: list(ast.literal_eval(hp[key + "_head_template"]))
            for target, key in zip(TARGET_NAMES, OUTPUT_NAMES)}


def fit_model(config, data, directory, *, variant=None, low_memory=False):
    """Fit one network with checkpoint, full history, physical predictions and metrics."""
    tf = runtime(config)
    from data_preprocessing.augmentation import augment_training_data, prepare_multitask_dicts
    from model_definitions.model_definitions import build_model, build_export_model, WarmupCosineDecay
    from training.train_best_tuner_trial import FixedTrialHP
    from training.training import NormalizedMaeSumCallback, _create_dataset, compute_inverse_variance_loss_weights
    from results_and_evaluations.prediction_utils import prediction_matrix
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    x, xv = data["X_train_features"], data["X_val_features"]
    yt, yv = prepare_multitask_dicts(data["y_train_targets"], data["y_val_targets"])
    aug = config["augmentation"]
    factor, noise = aug["factor"], aug["noise_level"]
    source_labels = {k: v.copy() for k, v in yt.items()}
    order = None
    no_noise = variant == "no_noise_augmentation"
    if low_memory or no_noise:
        from scripts.retrain_corrected_ablations import cached_augmentation
        if factor != 3 or noise != .05 or config["seed"] != 42:
            raise ValueError("Disk-backed paper copies require factor 3, noise .05 and seed 42.")
        xa, order = cached_augmentation(x, directory.parent / "augmented_train.npy")
        yt = {k: np.tile(v, factor)[order] for k, v in yt.items()}
    else:
        xa, yt = augment_training_data(x, yt, aug_factor=factor, noise_level=noise, seed=config["seed"])
    weights = compute_inverse_variance_loss_weights(yt, OUTPUT_NAMES)
    std = {k: float(np.std(yt[k], dtype=np.float64)) for k in OUTPUT_NAMES}
    steps = len(xa) // config["batch_size"]
    if steps < 1:
        raise ValueError("Batch size exceeds the augmented training sample count.")
    indices = list(range(3))
    if variant is None:
        model = build_model(
            input_shape=4000, hp=FixedTrialHP(config["hyperparameters"]),
            loss_weights=weights, total_steps=steps * config["epochs"],
            use_lr_schedule=True, architecture=config["architecture"],
            jit_compile=False, batch_size=config["batch_size"],
        )
        if config["hyperparameters"] == load_config()["hyperparameters"] and model.count_params() != 530275:
            raise ValueError("Paper architecture differs from the declared 530,275 parameters.")
    else:
        from scripts.retrain_corrected_ablations import build_variant
        model, indices = build_variant(variant, config["epochs"], steps, weights, batch_size=config["batch_size"])
    keys = [OUTPUT_NAMES[i] for i in indices]
    callbacks = []
    if variant is not None:
        from model_definitions.model_definitions import _build_optimizer
        optimizer = _build_optimizer(hp=FixedTrialHP(config["hyperparameters"]),
                                     total_steps=steps * config["epochs"], batch_size=config["batch_size"])
        model.compile(
            optimizer=optimizer,
            loss={k: tf.keras.losses.Huber(delta=1.) for k in keys},
            loss_weights={k: weights[k] if len(keys) > 1 else 1. for k in keys},
            metrics={k: [tf.keras.metrics.MeanAbsoluteError(name="mae")] for k in keys},
        )
        if len(keys) == 1:
            from scripts.retrain_corrected_ablations import AliasSingleMetric
            callbacks.append(AliasSingleMetric(keys[0]))
    if low_memory or no_noise:
        from scripts.retrain_corrected_ablations import indexed_dataset
        train = indexed_dataset(x if no_noise else xa, source_labels, order, keys,
                                repeat_original=no_noise, batch_size=config["batch_size"])
    else:
        train = _create_dataset(xa, {k: yt[k] for k in keys}, config["batch_size"], shuffle=True, cache=False)
    validation = _create_dataset(xv, {k: yv[k] for k in keys}, config["batch_size"], shuffle=False, cache=False)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    train, validation = train.with_options(options), validation.with_options(options)
    callbacks += [
        NormalizedMaeSumCallback(std, keys),
        tf.keras.callbacks.ModelCheckpoint(str(directory / "best.weights.h5"),
            monitor="val_normalized_mae_sum", mode="min", save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.CSVLogger(str(directory / "history.csv")),
    ]
    history = model.fit(train, validation_data=validation, epochs=config["epochs"], callbacks=callbacks, verbose=2)
    raw = {k: [float(v) for v in values] for k, values in history.history.items()}
    write_json(directory / "history.json", raw)
    model.load_weights(directory / "best.weights.h5")
    model_name = "retrained_paper_model.keras" if variant is None else "model.keras"
    build_export_model(model).save(directory / model_name)
    prediction = prediction_matrix(model.predict(data["X_test_features"], batch_size=config["batch_size"], verbose=0), keys).astype(np.float64)
    prediction = prediction * data["label_robust_scale"][indices].astype(np.float64) + data["label_robust_center"][indices].astype(np.float64)
    truth = physical_labels(data)[:, indices]
    write_predictions(directory / "test_predictions.npz", data["test_ids"], truth, prediction,
                      dataset_id=str(data["dataset_id"].item()), target_names=tuple(TARGET_NAMES[i] for i in indices))
    mae = np.mean(abs(prediction - truth), axis=0)
    report = {
        **config, "trainable_parameters": int(model.count_params()), "params": int(model.count_params()),
        "heads": {TARGET_NAMES[i]: _heads(config["hyperparameters"])[TARGET_NAMES[i]] for i in indices},
        "steps_per_epoch": steps, "original_train_samples": len(x), "augmented_train_samples": len(xa),
        "validation_samples": len(xv), "loss_weights": {k: weights[k] if len(keys) > 1 else 1. for k in keys},
        "target_standard_deviations": std,
        "selected_epoch": int(np.nanargmin(raw["val_normalized_mae_sum"])) + 1,
        "best_validation_score": float(min(raw["val_normalized_mae_sum"])),
        "test_mae": dict(zip((TARGET_NAMES[i] for i in indices), mae.tolist())),
        "dataset_id": str(data["dataset_id"].item()), "model_sha256": sha256(directory / model_name),
        "variant": variant, "software_versions": software_versions(),
        "reference_label_convention": "inverse_scaled_float64_v1",
    }
    if variant is not None:
        report.update(noise_augmentation=not no_noise, training_rows_per_epoch=len(xa),
                      original_training_objects=len(x), sample_exposure="matched augmented-example and optimizer-step budgets")
    write_json(directory / ("run_config.json" if variant is None else "metrics.json"), report)
    del model, xa, train, validation
    tf.keras.backend.clear_session()
    gc.collect()
    return report


def train_primary(run, config, data):
    outputs = ["retrained_paper_model.keras", "best.weights.h5", "history.csv", "history.json",
               "test_predictions.npz", "run_config.json"]
    with run.stage("train", outputs):
        report = fit_model(config, data, run.path)
        write_json(run.path / "software_versions.json", software_versions())
        write_json(run.path / "source_provenance.json", source_provenance())
    return report


def train_ablations(run, config, data, *, variants=None, low_memory=False):
    from scripts.retrain_corrected_ablations import VARIANTS
    if config["hyperparameters"] != load_config()["hyperparameters"]:
        raise ValueError("These controlled ablations support the paper's trial-099 architecture only.")
    variants = list(VARIANTS) if not variants or variants == ["all"] else list(variants)
    if len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError("Unknown or repeated ablation variant.")
    # Check all requested collisions before starting the first expensive fit.
    if any((run.path / "ablations" / v).exists() for v in variants):
        raise FileExistsError("Requested ablation output exists; use new variants or a fresh run.")
    results = []
    for variant in variants:
        with run.stage("ablation:" + variant, ["ablations/" + variant]):
            results.append(fit_model(config, data, run.path / "ablations" / variant,
                                     variant=variant, low_memory=low_memory))
    return results


def train_baselines(run, config, data, *, models=("ols", "ridge", "cnn", "dnn"), allow_cpu=False, pretrain_epochs=20):
    models = list(models)
    if not models or len(set(models)) != len(models) or any(m not in ("ols", "ridge", "cnn", "dnn", "transformer") for m in models):
        raise ValueError("Choose distinct supported baseline models.")
    if any((run.path / "baselines" / m).exists() for m in models):
        raise FileExistsError("Requested baseline outputs already exist; choose new models or a fresh run.")
    if pretrain_epochs < 1:
        raise ValueError("Pretraining epochs must be positive.")
    if "transformer" in models:
        try:
            import torch
        except ImportError as error:
            raise ImportError("Install requirements-torch.txt for the Transformer baseline.") from error
        if not torch.cuda.is_available() and not allow_cpu:
            raise RuntimeError("Transformer training requires CUDA, or explicit --allow-cpu for debugging.")
    from .predictions import physical_predictions
    from training.train_paper_hyperparams import _target_scalers_from_npz
    results = []
    for name in models:
        folder = run.path / "baselines" / name
        with run.stage("baseline:" + name, ["baselines/" + name]):
            folder.mkdir(parents=True)
            if name == "transformer":
                from finetuning.train_spectral_transformer import parse_args, train_spectral_transformer
                from .datasets import COMPACT_NAME
                info = json.loads((run.path / "data_reference.json").read_text())
                args = parse_args([])
                args.data = (run.path / info["cache_relative"] / COMPACT_NAME).resolve()
                args.output_dir = folder
                args.epochs, args.batch_size, args.seed = config["epochs"], config["batch_size"], config["seed"]
                args.aug_factor = config["augmentation"]["factor"]
                args.noise_level = config["augmentation"]["noise_level"]
                args.allow_cpu = allow_cpu
                args.workers = 0
                train_spectral_transformer(args)
                report = json.loads((folder / "test_metrics.json").read_text())
                report["selection_metric_version"] = "object_weighted_huber_v2"
            elif name in ("ols", "ridge"):
                import joblib
                from sklearn.linear_model import LinearRegression, RidgeCV
                estimator = LinearRegression() if name == "ols" else RidgeCV(alphas=np.logspace(-3, 3, 13))
                estimator.fit(data["X_train_features"], data["y_train_targets"])
                prediction = physical_predictions(estimator.predict(data["X_test_features"]), data)
                joblib.dump(estimator, folder / "model.joblib")
                write_predictions(folder / "test_predictions.npz", data["test_ids"], physical_labels(data), prediction,
                                  dataset_id=str(data["dataset_id"].item()))
                report = {"method": name}
            else:
                tf = runtime(config)
                from results_and_evaluations.model_benchmarks import run_neural_benchmarks
                rows = run_neural_benchmarks(
                    data["X_train_features"], data["y_train_targets"],
                    data["X_val_features"], data["y_val_targets"],
                    data["X_test_features"], data["y_test_targets"],
                    _target_scalers_from_npz(data), TARGET_NAMES,
                    include_cnn=name == "cnn", include_li_dnn=name == "dnn",
                    aug_factor=config["augmentation"]["factor"], noise_level=config["augmentation"]["noise_level"],
                    seed=config["seed"], epochs=config["epochs"], batch_size=config["batch_size"],
                    li_pretrain_epochs=pretrain_epochs, model_output_dir=folder,
                )
                report = rows[-1]
                if name == "cnn":
                    model = tf.keras.models.load_model(folder / "fabbro_cnn.keras", compile=False)
                    scaled = model.predict(data["X_test_features"][..., None], batch_size=config["batch_size"], verbose=0)
                    del model
                else:
                    columns = []
                    for target in ("teff", "feh", "logg"):
                        model = tf.keras.models.load_model(folder / f"li_dnn_{target}.keras", compile=False)
                        columns.append(model.predict(data["X_test_features"], batch_size=config["batch_size"], verbose=0).ravel())
                        del model
                    scaled = np.column_stack(columns)
                prediction = physical_predictions(scaled, data)
                write_predictions(folder / "test_predictions.npz", data["test_ids"], physical_labels(data), prediction,
                                  dataset_id=str(data["dataset_id"].item()))
                tf.keras.backend.clear_session()
            from .predictions import read_predictions
            archive = read_predictions(folder / "test_predictions.npz")
            report.update(
                model=name, seed=config["seed"], epochs=config["epochs"],
                batch_size=config["batch_size"], dataset_id=str(data["dataset_id"].item()),
                test_mae=dict(zip(TARGET_NAMES, np.mean(abs(archive["y_pred"] - archive["y_true"]), axis=0).tolist())),
            )
            # Benchmark rows use NaN for diagnostics that were not measured.
            report = {k: None if isinstance(v, (float, np.floating)) and np.isnan(v) else v
                      for k, v in report.items()}
            write_json(folder / "metrics.json", report)
            write_json(folder / "configuration.json", {**config, "model": name, "pretrain_epochs":pretrain_epochs})
            results.append(report)
        gc.collect()
    return results


def tune_primary(run, config, data, *, max_trials=100, search_epochs=30):
    if max_trials < 1 or search_epochs < 1:
        raise ValueError("Trial and search-epoch counts must be positive.")
    tf = runtime(config)
    from data_preprocessing.augmentation import augment_training_data, prepare_multitask_dicts
    from training.training import tune_and_train, compute_inverse_variance_loss_weights
    from .config import load_tuning_history
    from .predictions import physical_predictions
    import pandas as pd
    outputs = ["retrained_paper_model.keras", "best.weights.h5", "history.csv", "history.json",
               "test_predictions.npz", "run_config.json", "tuner", "tuning_history.json"]
    with run.stage("tune", outputs):
        yt, yv = prepare_multitask_dicts(data["y_train_targets"], data["y_val_targets"])
        x, yt = augment_training_data(data["X_train_features"], yt,
            aug_factor=config["augmentation"]["factor"], noise_level=config["augmentation"]["noise_level"], seed=config["seed"])
        if len(x) < config["batch_size"]:
            raise ValueError("Batch size exceeds augmented training size.")
        result = tune_and_train(
            x, yt, data["X_val_features"], yv, tuner_directory=str(run.path / "tuner"),
            tuner_project_name="search", model_checkpoint_path=str(run.path / "retrained_paper_model.keras"),
            final_model_path=str(run.path / "retrained_paper_model.keras"),
            tuner_max_trials=max_trials, search_epochs=search_epochs,
            search_batch_size=config["batch_size"], final_train_epochs=config["epochs"], jit_compile=False,
        )
        model, history = result["model"], result["history"].history
        model.save_weights(run.path / "best.weights.h5")
        frame = pd.DataFrame(history)
        frame.index.name = "epoch"
        frame.to_csv(run.path / "history.csv")
        write_json(run.path / "history.json", {k: [float(v) for v in values] for k, values in history.items()})
        trials = load_tuning_history(run.path / "tuner/search")
        write_json(run.path / "tuning_history.json", {"schema_version": 1, "trials": trials})
        winner = min((t for t in trials if t["status"] == "COMPLETED"), key=lambda t: t["score"])
        prediction = physical_predictions(model.predict(data["X_test_features"], verbose=0), data)
        truth = physical_labels(data)
        write_predictions(run.path / "test_predictions.npz", data["test_ids"], truth, prediction,
                          dataset_id=str(data["dataset_id"].item()))
        hp = result["best_hp"].values
        report = {**config, "profile": "tuned", "hyperparameters": hp,
                  "source_trial_id": winner["trial_id"], "source_trial_score": winner["score"],
                  "heads": _heads(hp), "trainable_parameters": int(model.count_params()),
                  "selected_epoch": int(np.nanargmin(history["val_normalized_mae_sum"])) + 1,
                  "best_validation_score": float(min(history["val_normalized_mae_sum"])),
                  "test_mae": dict(zip(TARGET_NAMES, np.mean(abs(prediction-truth),axis=0).tolist())),
                  "model_sha256": sha256(run.path / "retrained_paper_model.keras"),
                  "loss_weights": compute_inverse_variance_loss_weights(yt, OUTPUT_NAMES),
                  "target_standard_deviations": {k:float(np.std(yt[k],dtype=np.float64)) for k in OUTPUT_NAMES},
                  "dataset_id":str(data["dataset_id"].item())}
        write_json(run.path / "run_config.json", report)
        write_json(run.path / "source_provenance.json", source_provenance())
    tf.keras.backend.clear_session()
    return report
