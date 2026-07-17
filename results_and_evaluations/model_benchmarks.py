"""Benchmark baselines for comparison with the paper model.

The public release keeps the two classical baselines (`OLS`, `Ridge`) and
adds two optional neural baselines aligned with the comparisons described in
the paper:

- a StarNet-style 1D CNN inspired by Fabbro et al. (2017),
- a deep feedforward DNN following the layer-wise autoencoder pretraining and
  layer widths used by Li et al. (2017).

Both neural baselines use the same train/validation/test split as the paper
model and can use the same Gaussian-noise augmentation settings.
"""

from __future__ import annotations

import os
import tempfile
import time
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score

from data_preprocessing.augmentation import augment_training_data, prepare_multitask_dicts
from model_definitions.model_definitions import DEFAULT_COSINE_ALPHA, DEFAULT_HUBER_DELTA, WARMUP_FRACTION, WarmupCosineDecay

_MACRO_SCALES: dict[str, float] = {
    "Teff": 100.0,
    "[Fe/H]": 0.20,
    "log g": 0.20,
}
_TARGET_KEYS = ("teff_output", "feh_output", "logg_output")
_TARGET_KEY_TO_INDEX = {key: idx for idx, key in enumerate(_TARGET_KEYS)}
_LI_HIDDEN_UNITS = (1000, 500, 100, 30)
_LI_PRETRAIN_UNITS = _LI_HIDDEN_UNITS + (1,)
_LI_AE_MAX_EPOCHS = 20
_LI_AE_LEARNING_RATE = 1e-3
_LI_SUPERVISED_LEARNING_RATE = 1e-4
_LI_ADAMW_WEIGHT_DECAY = 3.958271671901557e-05


def _tf():
    import tensorflow as tf

    return tf


def _unscale(y_scaled: np.ndarray, target_scalers: Sequence) -> np.ndarray:
    """Inverse-transform ``(N, n_targets)`` scaled targets to physical space."""
    y = np.empty_like(y_scaled, dtype=np.float64)
    for idx, scaler in enumerate(target_scalers):
        y[:, idx] = scaler.inverse_transform(y_scaled[:, idx].reshape(-1, 1)).ravel()
    return y


def _compute_metrics(
    y_pred_scaled: np.ndarray,
    y_true_scaled: np.ndarray,
    target_scalers: Sequence,
    target_names: Sequence[str],
) -> dict[str, float]:
    """Per-target MAE, R2, and macro MAE in unscaled space."""
    y_pred = _unscale(y_pred_scaled, target_scalers)
    y_true = _unscale(y_true_scaled, target_scalers)

    metrics: dict[str, float] = {}
    macro_sum = 0.0
    for idx, name in enumerate(target_names):
        mae = mean_absolute_error(y_true[:, idx], y_pred[:, idx])
        r2 = r2_score(y_true[:, idx], y_pred[:, idx])
        metrics[f"MAE_{name}"] = mae
        metrics[f"R2_{name}"] = r2
        macro_sum += mae / _MACRO_SCALES.get(name, 1.0)
    metrics["macro_mae"] = macro_sum / len(target_names)
    return metrics


def _make_row(
    method: str,
    mode: str,
    metrics: dict[str, float],
    fit_time: float,
    *,
    params: int | float = float("nan"),
    train_samples: int | float = float("nan"),
    epochs: int | float = float("nan"),
    pretrain_epochs: int | float = float("nan"),
    weight_decay: float = float("nan"),
    ae_l2_weight_decay: float = float("nan"),
    aug_factor: int | float = float("nan"),
) -> dict[str, object]:
    return {
        "method": method,
        "mode": mode,
        **metrics,
        "fit_time_s": round(fit_time, 2),
        "params": params,
        "train_samples": train_samples,
        "epochs": epochs,
        "pretrain_epochs": pretrain_epochs,
        "weight_decay": weight_decay,
        "ae_l2_weight_decay": ae_l2_weight_decay,
        "aug_factor": aug_factor,
    }


def _show_progress(results: list[dict[str, object]]) -> None:
    """Print the current benchmark summary as a compact text table."""
    if not results:
        return

    mae_keys = [key for key in results[0] if key.startswith("MAE_")]
    header = f"  {'Method':<28}"
    for key in mae_keys:
        header += f" {key:>12}"
    header += f" {'macro_mae':>10} {'time':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in results:
        line = f"  {row['method']:<28}"
        for key in mae_keys:
            line += f" {row[key]:>12.3f}"
        line += f" {row['macro_mae']:>10.4f} {row['fit_time_s']:>6.1f}s"
        print(line)


def _fit_evaluate_multi(
    model,
    method_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scalers: Sequence,
    target_names: Sequence[str],
) -> dict[str, object]:
    """Fit one multi-output model and evaluate it on the test split."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

    predictions = model.predict(X_test)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    metrics = _compute_metrics(predictions, y_test, target_scalers, target_names)
    return _make_row(method_name, "multi-output", metrics, fit_time)


def run_linear_benchmarks(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scalers: Sequence,
    target_names: Sequence[str],
    _prior_results: list[dict] | None = None,
) -> list[dict[str, object]]:
    """Fit the public-release linear baselines and return result dicts."""
    del X_val, y_val

    results: list[dict[str, object]] = list(_prior_results or [])
    args = (X_train, y_train, X_test, y_test, target_scalers, target_names)

    print("  OLS ...")
    results.append(_fit_evaluate_multi(LinearRegression(), "OLS", *args))
    _show_progress(results)

    print("  Ridge ...")
    alphas = np.logspace(-3, 3, 13)
    results.append(_fit_evaluate_multi(RidgeCV(alphas=alphas), "Ridge", *args))
    _show_progress(results)

    return results


def _stack_multitask_targets(y_dict: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([np.asarray(y_dict[key], dtype=np.float32) for key in _TARGET_KEYS])


def _cast_target_dicts_to_float32(
    y_train_dict: dict[str, np.ndarray],
    y_val_dict: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return (
        {key: np.asarray(values, dtype=np.float32) for key, values in y_train_dict.items()},
        {key: np.asarray(values, dtype=np.float32) for key, values in y_val_dict.items()},
    )


def _build_fabbro_cnn(
    input_dim: int,
    *,
    kernel_size_1: int = 8,
    kernel_size_2: int = 8,
    pooling_size: int = 4,
    learning_rate: float = 1e-3,
):
    """Build a StarNet-style CNN inspired by Fabbro et al. (2017).

    The source paper specifies two 1D convolutional layers with 4 and 16
    filters, a max-pooling layer with window length 4, and dense layers with
    256, 128, and 3 nodes. The kernel lengths are exposed here because the
    figure caption lists them as tuned hyperparameters but does not pin a
    single public value in the schematic.
    """
    tf = _tf()
    inputs = tf.keras.layers.Input(shape=(input_dim, 1), name="stellar_spectrum")
    x = tf.keras.layers.Conv1D(4, kernel_size_1, padding="same", activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.Conv1D(16, kernel_size_2, padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=pooling_size, name="maxpool")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    outputs = tf.keras.layers.Dense(3, activation="linear", dtype="float32", name="predictions")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="fabbro_starnet_like")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def _build_li_single_target_dnn(
    input_dim: int,
    *,
    pretrained_encoder_weights: Sequence[Sequence[np.ndarray]],
    total_steps: int,
):
    """Build a Li et al. (2017)-style single-target DNN.

    Li et al. describe a six-layer network with neuron counts
    ``(3821, 1000, 500, 100, 30, 1)`` for single-parameter estimation.
    This public benchmark adapts the input dimensionality to the current SDSS
    DR12 grid, initializes all five connections with greedy stacked-autoencoder
    pretraining, and then fine-tunes the DNN end-to-end on labeled spectra. Li
    et al.'s initialization algorithm includes the final ``30 -> 1`` edge,
    while the supervised DNN output itself remains linear.
    """
    tf = _tf()
    inputs = tf.keras.layers.Input(shape=(input_dim,), name="stellar_spectrum")
    x = inputs
    for idx, units in enumerate(_LI_HIDDEN_UNITS):
        x = tf.keras.layers.Dense(
            units,
            activation="sigmoid",
            name=f"dense_{idx}_{units}",
        )(x)
    outputs = tf.keras.layers.Dense(
        1,
        activation="linear",
        dtype="float32",
        name="prediction",
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="li_dnn_single_target")
    for idx, weights in enumerate(pretrained_encoder_weights[:-1]):
        model.get_layer(f"dense_{idx}_{_LI_HIDDEN_UNITS[idx]}").set_weights(weights)
    model.get_layer("prediction").set_weights(pretrained_encoder_weights[-1])
    warmup_steps = int(total_steps * WARMUP_FRACTION)
    learning_rate = WarmupCosineDecay(
        _LI_SUPERVISED_LEARNING_RATE,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        alpha=DEFAULT_COSINE_ALPHA,
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=_LI_ADAMW_WEIGHT_DECAY,
            global_clipnorm=5.0,
        ),
        loss=tf.keras.losses.Huber(delta=DEFAULT_HUBER_DELTA),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def _build_li_autoencoder(
    input_dim: int,
    encoder_units: int,
    *,
    weight_decay: float,
):
    """Build one sigmoid autoencoder used for Li-style greedy pretraining."""
    tf = _tf()
    regularizer = tf.keras.regularizers.l2(weight_decay)
    inputs = tf.keras.layers.Input(shape=(input_dim,), name="autoencoder_input")
    encoded = tf.keras.layers.Dense(
        encoder_units,
        activation="sigmoid",
        kernel_regularizer=regularizer,
        name="encoder",
    )(inputs)
    reconstructed = tf.keras.layers.Dense(
        input_dim,
        activation="linear",
        kernel_regularizer=regularizer,
        name="reconstruction",
    )(encoded)
    autoencoder = tf.keras.Model(inputs, reconstructed, name=f"li_autoencoder_{input_dim}_{encoder_units}")
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=_LI_AE_LEARNING_RATE), loss="mse")
    return autoencoder


def _pretrain_li_encoders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    weight_decay: float = 1e-4,
) -> list[list[np.ndarray]]:
    """Greedily pretrain Li DNN encoders on spectra, without target labels.

    Each autoencoder reconstructs the representation output by the preceding
    encoder. Its learned encoder weights seed the corresponding supervised DNN
    layer. Validation spectra select the reconstruction checkpoint but never
    contribute to gradient updates.
    """
    tf = _tf()
    train_representation = np.asarray(X_train, dtype=np.float32)
    val_representation = np.asarray(X_val, dtype=np.float32)
    encoder_weights: list[list[np.ndarray]] = []

    for layer_idx, units in enumerate(_LI_PRETRAIN_UNITS, start=1):
        print(f"    Li autoencoder pretraining layer {layer_idx}: {train_representation.shape[1]} -> {units}")
        autoencoder = _build_li_autoencoder(
            train_representation.shape[1],
            units,
            weight_decay=weight_decay,
        )
        with tempfile.TemporaryDirectory(prefix=f"li_pretrain_{layer_idx}_") as tmpdir:
            weights_path = os.path.join(tmpdir, "best.weights.h5")
            autoencoder.fit(
                train_representation,
                train_representation,
                validation_data=(val_representation, val_representation),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[_make_checkpoint_callback(weights_path)],
                verbose=1,
            )
            autoencoder.load_weights(weights_path)
            encoder = tf.keras.Model(autoencoder.input, autoencoder.get_layer("encoder").output)
            encoder_weights.append([weight.copy() for weight in autoencoder.get_layer("encoder").get_weights()])
            train_representation = encoder.predict(train_representation, batch_size=batch_size, verbose=0)
            val_representation = encoder.predict(val_representation, batch_size=batch_size, verbose=0)

    return encoder_weights


def _make_checkpoint_callback(weights_path: str, *, monitor: str = "val_loss") -> object:
    tf = _tf()
    return tf.keras.callbacks.ModelCheckpoint(
        weights_path,
        monitor=monitor,
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )


def _train_fabbro_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scalers: Sequence,
    target_names: Sequence[str],
    *,
    epochs: int,
    batch_size: int,
    model_output_dir: Path | None = None,
) -> dict[str, object]:
    model = _build_fabbro_cnn(X_train.shape[1])
    with tempfile.TemporaryDirectory(prefix="benchmark_cnn_") as tmpdir:
        weights_path = os.path.join(tmpdir, "best.weights.h5")
        callbacks = [_make_checkpoint_callback(weights_path)]

        t0 = time.perf_counter()
        model.fit(
            X_train[..., np.newaxis],
            y_train,
            validation_data=(X_val[..., np.newaxis], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        fit_time = time.perf_counter() - t0
        model.load_weights(weights_path)
        if model_output_dir is not None:
            model.save(model_output_dir / "fabbro_cnn.keras")

        predictions = np.asarray(
            model.predict(X_test[..., np.newaxis], batch_size=batch_size, verbose=0),
            dtype=np.float32,
        )
    metrics = _compute_metrics(predictions, y_test, target_scalers, target_names)
    return _make_row(
        "CNN (Fabbro et al. 2017)",
        "multi-output",
        metrics,
        fit_time,
        params=int(model.count_params()),
        train_samples=int(X_train.shape[0]),
        epochs=epochs,
    )


def _train_li_dnn_ensemble(
    X_train: np.ndarray,
    y_train_dict: dict[str, np.ndarray],
    X_val: np.ndarray,
    y_val_dict: dict[str, np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scalers: Sequence,
    target_names: Sequence[str],
    *,
    epochs: int,
    batch_size: int,
    pretrain_epochs: int,
    weight_decay: float,
    model_output_dir: Path | None = None,
) -> dict[str, object]:
    predictions_scaled = np.empty_like(y_test, dtype=np.float32)
    total_params = 0
    actual_pretrain_epochs = max(1, min(_LI_AE_MAX_EPOCHS, pretrain_epochs))
    steps_per_epoch = max(1, int(np.ceil(X_train.shape[0] / batch_size)))
    total_steps = max(1, steps_per_epoch * epochs)

    t0 = time.perf_counter()
    pretrained_encoder_weights = _pretrain_li_encoders(
        X_train,
        X_val,
        epochs=actual_pretrain_epochs,
        batch_size=batch_size,
        weight_decay=weight_decay,
    )
    for target_key in _TARGET_KEYS:
        model = _build_li_single_target_dnn(
            X_train.shape[1],
            pretrained_encoder_weights=pretrained_encoder_weights,
            total_steps=total_steps,
        )
        with tempfile.TemporaryDirectory(prefix=f"benchmark_{target_key}_") as tmpdir:
            weights_path = os.path.join(tmpdir, "best.weights.h5")
            callbacks = [_make_checkpoint_callback(weights_path, monitor="val_mae")]
            model.fit(
                X_train,
                y_train_dict[target_key],
                validation_data=(X_val, y_val_dict[target_key]),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
            )
            model.load_weights(weights_path)
        if model_output_dir is not None:
            model.save(model_output_dir / f"li_dnn_{target_key.removesuffix('_output')}.keras")
        target_idx = _TARGET_KEY_TO_INDEX[target_key]
        predictions_scaled[:, target_idx] = model.predict(
            X_test,
            batch_size=batch_size,
            verbose=0,
        ).reshape(-1)
        total_params += int(model.count_params())
    fit_time = time.perf_counter() - t0

    metrics = _compute_metrics(predictions_scaled, y_test, target_scalers, target_names)
    return _make_row(
        "DNN (Li et al. 2017)",
        "single-target ensemble",
        metrics,
        fit_time,
        params=total_params,
        train_samples=int(X_train.shape[0]),
        epochs=epochs,
        pretrain_epochs=actual_pretrain_epochs,
        weight_decay=_LI_ADAMW_WEIGHT_DECAY,
        ae_l2_weight_decay=weight_decay,
    )


def run_neural_benchmarks(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scalers: Sequence,
    target_names: Sequence[str],
    *,
    include_cnn: bool = True,
    include_li_dnn: bool = True,
    aug_factor: int = 3,
    noise_level: float = 0.05,
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 256,
    li_pretrain_epochs: int = _LI_AE_MAX_EPOCHS,
    li_weight_decay: float = 1e-4,
    model_output_dir: str | Path | None = None,
    _prior_results: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Fit neural baselines on the same split and augmentation regime as the paper model."""
    results: list[dict[str, object]] = list(_prior_results or [])
    output_dir = Path(model_output_dir) if model_output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if include_cnn or include_li_dnn:
        tf = _tf()
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    y_train_dict, y_val_dict = prepare_multitask_dicts(y_train, y_val)
    y_train_dict, y_val_dict = _cast_target_dicts_to_float32(y_train_dict, y_val_dict)

    augmented_X_train: np.ndarray | None = None
    augmented_y_train_dict: dict[str, np.ndarray] | None = None
    if include_cnn or include_li_dnn:
        augmented_X_train = np.asarray(X_train, dtype=np.float32)
        augmented_y_train_dict = {key: values.copy() for key, values in y_train_dict.items()}
        if aug_factor > 1:
            augmented_X_train, augmented_y_train_dict = augment_training_data(
                augmented_X_train,
                augmented_y_train_dict,
                aug_factor=aug_factor,
                noise_level=noise_level,
                seed=seed,
            )

    if include_cnn:
        assert augmented_X_train is not None and augmented_y_train_dict is not None
        cnn_y_train = _stack_multitask_targets(augmented_y_train_dict)
        print("  CNN (Fabbro et al. 2017) ...")
        cnn_row = _train_fabbro_cnn(
            augmented_X_train,
            cnn_y_train,
            np.asarray(X_val, dtype=np.float32),
            np.asarray(y_val, dtype=np.float32),
            np.asarray(X_test, dtype=np.float32),
            np.asarray(y_test, dtype=np.float32),
            target_scalers,
            target_names,
            epochs=epochs,
            batch_size=batch_size,
            model_output_dir=output_dir,
        )
        cnn_row["aug_factor"] = aug_factor
        results.append(cnn_row)
        _show_progress(results)

    if include_li_dnn:
        assert augmented_X_train is not None and augmented_y_train_dict is not None
        print("  DNN (Li et al. 2017) ...")
        dnn_row = _train_li_dnn_ensemble(
            augmented_X_train,
            augmented_y_train_dict,
            np.asarray(X_val, dtype=np.float32),
            y_val_dict,
            np.asarray(X_test, dtype=np.float32),
            np.asarray(y_test, dtype=np.float32),
            target_scalers,
            target_names,
            epochs=epochs,
            batch_size=batch_size,
            pretrain_epochs=li_pretrain_epochs,
            weight_decay=li_weight_decay,
            model_output_dir=output_dir,
        )
        dnn_row["aug_factor"] = aug_factor
        results.append(dnn_row)
        _show_progress(results)

    return results


def add_external_result(
    results_df: pd.DataFrame,
    method_name: str,
    predictions_scaled: np.ndarray,
    y_test: np.ndarray,
    target_scalers: Sequence,
    target_names: Sequence[str],
    *,
    mode: str = "multi-output",
    fit_time: float = float("nan"),
) -> pd.DataFrame:
    """Inject results from an external model into the benchmark table."""
    metrics = _compute_metrics(predictions_scaled, y_test, target_scalers, target_names)
    row = _make_row(method_name, mode, metrics, fit_time)
    return pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)


def run_all_benchmarks(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scalers: Sequence,
    target_names: Sequence[str],
    *,
    include_neural_baselines: bool = False,
    include_cnn: bool = True,
    include_li_dnn: bool = True,
    aug_factor: int = 3,
    noise_level: float = 0.05,
    seed: int = 42,
    neural_epochs: int = 30,
    neural_batch_size: int = 256,
    li_pretrain_epochs: int = _LI_AE_MAX_EPOCHS,
    li_weight_decay: float = 1e-4,
    model_output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run the benchmark set and optionally save fitted neural models."""
    print("=" * 60)
    print("Running classical benchmarks: OLS and Ridge")
    if include_neural_baselines:
        print("Running neural baselines: StarNet-style CNN and Li-style DNN")
    print("=" * 60)
    results = run_linear_benchmarks(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        target_scalers,
        target_names,
    )
    if include_neural_baselines:
        results = run_neural_benchmarks(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            target_scalers,
            target_names,
            include_cnn=include_cnn,
            include_li_dnn=include_li_dnn,
            aug_factor=aug_factor,
            noise_level=noise_level,
            seed=seed,
            epochs=neural_epochs,
            batch_size=neural_batch_size,
            li_pretrain_epochs=li_pretrain_epochs,
            li_weight_decay=li_weight_decay,
            model_output_dir=model_output_dir,
            _prior_results=results,
        )
    return pd.DataFrame(results)


def print_benchmark_table(df: pd.DataFrame) -> None:
    """Print a neatly formatted benchmark results table."""
    mae_cols = [column for column in df.columns if column.startswith("MAE_")]

    header = f"{'Method':<28}"
    for column in mae_cols:
        header += f" {column:>12}"
    header += f" {'macro_mae':>10} {'time':>7}"

    print()
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, row in df.iterrows():
        line = f"  {row['method']:<28}"
        for column in mae_cols:
            line += f" {row[column]:>12.3f}"
        line += f" {row['macro_mae']:>10.4f} {row['fit_time_s']:>6.1f}s"
        print(line)
