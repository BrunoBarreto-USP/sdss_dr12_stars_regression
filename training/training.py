"""Training helpers: runtime config, search, fit, and history plotting.

Supports an explicit Colab GPU path and an explicit local CPU-only path.
On-the-fly augmentation (Gaussian noise + neighbour mixup + wavelength masking).
EMA weights for stable validation. Optional augmentation ramp schedule.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback, ModelCheckpoint

try:
    from model_definitions.model_definitions import (
        DEFAULT_ARCHITECTURE,
        DEFAULT_COSINE_ALPHA,
        DEFAULT_TUNER_PROJECT_NAME,
        WARMUP_FRACTION,
        build_bayesian_tuner,
        build_export_model,
        build_model,
        compile_huber_finetune,
    )
except ImportError:
    from model_definitions.model_definitions import (
        DEFAULT_ARCHITECTURE,
        DEFAULT_COSINE_ALPHA,
        DEFAULT_TUNER_PROJECT_NAME,
        WARMUP_FRACTION,
        build_bayesian_tuner,
        build_export_model,
        build_model,
        compile_huber_finetune,
    )


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


NORMALIZED_MAE_SUM_NAME = "normalized_mae_sum"
DEFAULT_TARGET_KEYS = ("teff_output", "feh_output", "logg_output")


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def _set_mixed_precision_policy(policy_name: str) -> str:
    tf.keras.mixed_precision.set_global_policy(policy_name)
    return tf.keras.mixed_precision.global_policy().name


def configure_gpu_runtime(
    *,
    require_gpu: bool = False,
    enable_mixed_precision: bool = True,
) -> dict[str, object]:
    """Configure TensorFlow for a single visible GPU runtime."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        policy = _set_mixed_precision_policy("float32")
        if require_gpu:
            raise RuntimeError(
                "No GPU detected. In Colab, enable a GPU runtime before training."
            )
        return {
            "device": "CPU",
            "gpu_name": None,
            "gpu_mem_mb": 0,
            "visible_gpu_count": 0,
            "mixed_precision_policy": policy,
        }

    selected_gpu = gpus[0]
    try:
        tf.config.set_visible_devices(selected_gpu, "GPU")
        tf.config.experimental.set_memory_growth(selected_gpu, True)
    except RuntimeError:
        # TensorFlow runtime was already initialized; continue with current visibility.
        pass

    try:
        details = tf.config.experimental.get_device_details(selected_gpu)
        gpu_name = details.get("device_name", "Unknown")
        gpu_mem_mb = int(details.get("memory_limit_mb", 0) or 0)
    except Exception:
        gpu_name, gpu_mem_mb = tf.test.gpu_device_name(), 0

    policy = _set_mixed_precision_policy(
        "mixed_float16" if enable_mixed_precision else "float32"
    )
    visible_gpu_count = len(tf.config.get_visible_devices("GPU"))
    return {
        "device": "GPU",
        "gpu_name": gpu_name,
        "gpu_mem_mb": gpu_mem_mb,
        "visible_gpu_count": visible_gpu_count,
        "mixed_precision_policy": policy,
    }


def configure_cpu_runtime() -> dict[str, object]:
    """Force TensorFlow to run on CPU only."""
    gpu_count = len(tf.config.list_physical_devices("GPU"))
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        if tf.config.get_visible_devices("GPU"):
            raise RuntimeError(
                "CPU-only mode must be configured before TensorFlow initializes the GPU runtime. "
                "Restart the kernel/runtime and run the setup cell first."
            ) from exc

    policy = _set_mixed_precision_policy("float32")
    return {
        "device": "CPU",
        "disabled_gpu_count": gpu_count,
        "cpu_count": os.cpu_count() or 1,
        "mixed_precision_policy": policy,
    }


def detect_gpu_batch_size(default: int = 256) -> int:
    """Return a batch size appropriate for the detected GPU.

    Falls back to *default* when no GPU is available.
    """
    gpus = tf.config.get_visible_devices("GPU")
    if not gpus:
        return default
    try:
        details = tf.config.experimental.get_device_details(gpus[0])
        name = details.get("device_name", "").lower()
        mem_mb = details.get("memory_limit_mb", 0)
    except Exception:
        name, mem_mb = "", 0

    if mem_mb > 60_000 or "a100" in name:        # A100 40/80 GB
        return 2048
    if mem_mb > 30_000 or "v100" in name:         # V100 32 GB
        return 1024
    if mem_mb > 14_000 or "t4" in name:           # T4 16 GB
        return 512
    return default


def detect_cpu_batch_size(default: int | None = None) -> int:
    """Return a conservative CPU-friendly batch size."""
    if default is not None:
        return int(default)

    cpu_count = os.cpu_count() or 1
    if cpu_count >= 16:
        return 128
    if cpu_count >= 8:
        return 64
    return 32


# ---------------------------------------------------------------------------
# On-the-fly augmentation helpers
# ---------------------------------------------------------------------------

def _create_dataset(
    X: np.ndarray,
    y_dict: dict[str, np.ndarray],
    batch_size: int,
    *,
    shuffle: bool = True,
    cache: bool = False,
) -> tf.data.Dataset:
    """Wrap arrays in an optimized tf.data pipeline (no augmentation)."""
    X = np.asarray(X, dtype=np.float32)
    y_dict = {key: np.asarray(values, dtype=np.float32) for key, values in y_dict.items()}
    dataset = tf.data.Dataset.from_tensor_slices((X, y_dict))
    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(X), 50_000))
    dataset = dataset.batch(batch_size, drop_remainder=shuffle)
    return dataset.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _resolve_target_keys(y_dict: dict[str, np.ndarray]) -> list[str]:
    preferred_order = [key for key in DEFAULT_TARGET_KEYS if key in y_dict]
    if len(preferred_order) == len(y_dict):
        return preferred_order
    return list(y_dict.keys())


def _compute_target_variances(
    y_dict: dict[str, np.ndarray],
    target_keys: Sequence[str],
) -> dict[str, float]:
    return {
        key: float(np.var(y_dict[key], dtype=np.float64))
        for key in target_keys
    }


def _compute_target_standard_deviations(
    y_dict: dict[str, np.ndarray],
    target_keys: Sequence[str],
) -> dict[str, float]:
    """Return target scales for dimensionless MAE model selection."""
    return {
        key: float(np.std(y_dict[key], dtype=np.float64))
        for key in target_keys
    }


def _score_from_eval_results(
    eval_results: dict[str, float],
    *,
    target_standard_deviations: dict[str, float],
    target_keys: Sequence[str],
) -> float:
    score = 0.0
    for key in target_keys:
        score += float(eval_results[f"{key}_mae"]) / target_standard_deviations[key]
    return score


def cast_target_dicts_to_float32(
    y_train_dict: dict[str, np.ndarray],
    y_val_dict: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    y_train_dict_float32 = {
        key: np.asarray(values, dtype=np.float32)
        for key, values in y_train_dict.items()
    }
    y_val_dict_float32 = {
        key: np.asarray(values, dtype=np.float32)
        for key, values in y_val_dict.items()
    }
    return y_train_dict_float32, y_val_dict_float32


class NormalizedMaeSumCallback(Callback):
    """Inject normalized train/validation MAE aggregates into epoch logs."""

    def __init__(
        self,
        target_standard_deviations: dict[str, float],
        target_keys: Sequence[str],
        metric_name: str = NORMALIZED_MAE_SUM_NAME,
    ) -> None:
        super().__init__()
        self.target_keys = list(target_keys)
        self.metric_name = metric_name
        self.target_standard_deviations = {
            key: max(float(target_standard_deviations[key]), 1e-8)
            for key in self.target_keys
        }

    def _compute(self, logs: dict[str, float], prefix: str) -> float | None:
        total = 0.0
        for key in self.target_keys:
            mae_value = logs.get(f"{prefix}{key}_mae")
            if mae_value is None:
                return None
            total += float(mae_value) / self.target_standard_deviations[key]
        return total

    def on_epoch_end(self, epoch, logs=None) -> None:
        if logs is None:
            return

        train_metric = self._compute(logs, prefix="")
        if train_metric is not None:
            logs[self.metric_name] = train_metric

        val_metric = self._compute(logs, prefix="val_")
        if val_metric is not None:
            logs[f"val_{self.metric_name}"] = val_metric


def compute_inverse_variance_loss_weights(
    y_train_dict_float32: dict[str, np.ndarray],
    target_keys: Sequence[str] = DEFAULT_TARGET_KEYS,
) -> dict[str, float]:
    target_variances = _compute_target_variances(y_train_dict_float32, target_keys)
    inverse_variance_weights = {
        key: 1.0 / max(variance, 1e-8)
        for key, variance in target_variances.items()
    }
    normalization = len(target_keys) / sum(inverse_variance_weights.values())
    loss_weights = {
        key: inverse_variance_weights[key] * normalization
        for key in target_keys
    }
    print("Scaled target variances:", target_variances)
    print("Normalized loss weights (sum matches output count):", loss_weights)
    return loss_weights


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def tune_and_train(
    X_train_features: np.ndarray,
    y_train_dict_float32: dict[str, np.ndarray],
    X_val_features: np.ndarray,
    y_val_dict_float32: dict[str, np.ndarray],
    *,
    tuner_directory: str = "keras_tuner",
    tuner_project_name: str | None = None,
    model_checkpoint_path: str = os.path.join(MODELS_DIR, "best_stellar_multitask_model.keras"),
    final_model_path: str = os.path.join(MODELS_DIR, "stellar_regression_model.keras"),
    architecture: str = DEFAULT_ARCHITECTURE,
    tuner_max_trials: int = 100,
    tuner_executions_per_trial: int = 1,
    search_epochs: int = 30,
    search_batch_size: int = 256,
    final_train_epochs: int | None = None,
    fine_tune_epochs: int = 0,
    fine_tune_learning_rate: float = 1e-4,
    jit_compile: bool = False,
    # Warm restarts
    warm_restart_epochs: int | None = None,
) -> dict[str, object]:
    """Run Keras Tuner search, then retrain the best architecture.

    After the hyperparameter search the best configuration is retrained
    for *final_train_epochs* epochs.  An optional Huber fine-tune stage
    can follow if *fine_tune_epochs* > 0.

    The tuner search uses linear warmup followed by monotonic cosine decay.
    The final training run uses the tuned peak LR with the same warmup
    fraction, then follows the same monotonic cosine decay as the search.

    ``warm_restart_epochs`` is retained for backward compatibility but is
    ignored; warm restarts are disabled for this project.
    """
    if tuner_project_name is None:
        tuner_project_name = DEFAULT_TUNER_PROJECT_NAME
    final_train_epochs = final_train_epochs if final_train_epochs is not None else search_epochs

    input_dim = X_train_features.shape[1]
    n_train = X_train_features.shape[0]
    target_keys = _resolve_target_keys(y_train_dict_float32)
    target_standard_deviations = _compute_target_standard_deviations(
        y_train_dict_float32,
        target_keys,
    )
    loss_weights = compute_inverse_variance_loss_weights(
        y_train_dict_float32,
        target_keys=target_keys,
    )

    weights_checkpoint_path = f"{os.path.splitext(model_checkpoint_path)[0]}.weights.h5"
    monitor = f"val_{NORMALIZED_MAE_SUM_NAME}"
    print(f"Training architecture: {architecture}")
    print(
        "Scaled target standard deviations for normalized MAE monitoring:",
        target_standard_deviations,
    )
    print(f"Monitoring '{monitor}' for tuning and checkpoint selection")
    print(f"Search epochs: {search_epochs}, Final train epochs: {final_train_epochs}")
    print(
        f"Search LR schedule: warmup ({WARMUP_FRACTION:.0%}) + cosine decay "
        f"(alpha={DEFAULT_COSINE_ALPHA:.2f})"
    )
    print(
        f"Final LR schedule: warmup ({WARMUP_FRACTION:.0%}) + cosine decay "
        f"(alpha={DEFAULT_COSINE_ALPHA:.2f})"
    )
    if fine_tune_epochs > 0:
        print(f"Huber fine-tune: {fine_tune_epochs} epochs, LR={fine_tune_learning_rate}")

    steps_per_epoch = int(n_train // search_batch_size)
    total_steps = steps_per_epoch * search_epochs

    if warm_restart_epochs is not None and warm_restart_epochs > 0:
        print(
            f"Warm restarts requested ({warm_restart_epochs} epochs) but ignored; "
            "using monotonic cosine decay instead."
        )

    # Build augmented training dataset
    search_dataset = _create_dataset(
        X_train_features, y_train_dict_float32,
        batch_size=search_batch_size, shuffle=True, cache=True,
    )
    val_dataset = _create_dataset(
        X_val_features, y_val_dict_float32,
        batch_size=search_batch_size, shuffle=False, cache=True,
    )

    model_build_kwargs = {
        "input_shape": input_dim,
        "num_outputs": len(target_keys),
        "loss_weights": loss_weights,
        "total_steps": total_steps,
        "cosine_alpha": DEFAULT_COSINE_ALPHA,
        "use_lr_schedule": True,
        "architecture": architecture,
        "jit_compile": jit_compile,
        "batch_size": search_batch_size,
    }

    if jit_compile:
        print("XLA JIT compilation: ENABLED")
    from model_definitions.model_definitions import REFERENCE_BATCH_SIZE
    lr_scale = (search_batch_size / REFERENCE_BATCH_SIZE) ** 0.5
    if lr_scale != 1.0:
        print(
            f"LR scaling: {lr_scale:.3f}x "
            f"(sqrt(batch {search_batch_size} / ref {REFERENCE_BATCH_SIZE}))"
        )

    print("--- Starting Bayesian search with Keras Tuner ---")
    tuner = build_bayesian_tuner(
        max_trials=tuner_max_trials,
        executions_per_trial=tuner_executions_per_trial,
        directory=tuner_directory,
        project_name=tuner_project_name,
        objective_name=monitor,
        **model_build_kwargs,
    )
    tuner.search(
        search_dataset,
        epochs=search_epochs,
        validation_data=val_dataset,
        callbacks=[
            NormalizedMaeSumCallback(target_standard_deviations, target_keys),
        ],
        verbose=1,
    )
    tuner.results_summary()

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    for name, value in best_hp.values.items():
        print(f"Hyperparameter {name}: {value}")

    # ---- Final training run ----
    final_total_steps = steps_per_epoch * final_train_epochs
    final_build_kwargs = dict(model_build_kwargs)
    final_build_kwargs["total_steps"] = final_total_steps

    print(f"\n{'=' * 60}")
    print("--- Final training run ---")
    print(f"{'=' * 60}")

    train_dataset = _create_dataset(
        X_train_features, y_train_dict_float32,
        batch_size=search_batch_size, shuffle=True, cache=True,
    )
    model = build_model(hp=best_hp, **final_build_kwargs)

    callbacks = [
        NormalizedMaeSumCallback(target_standard_deviations, target_keys),
        ModelCheckpoint(
            weights_checkpoint_path,
            monitor=monitor,
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_dataset,
        epochs=final_train_epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1,
    )

    if os.path.exists(weights_checkpoint_path):
        model.load_weights(weights_checkpoint_path)

    if fine_tune_epochs > 0:
        print(f"--- Huber fine-tune stage ({fine_tune_epochs} epochs) ---")
        ft_checkpoint_path = (
            f"{os.path.splitext(weights_checkpoint_path)[0]}_ft.weights.h5"
        )
        compile_huber_finetune(
            model,
            output_names=target_keys,
            loss_weights=loss_weights,
            learning_rate=fine_tune_learning_rate,
            jit_compile=jit_compile,
        )
        ft_callbacks = [
            NormalizedMaeSumCallback(target_standard_deviations, target_keys),
            ModelCheckpoint(
                ft_checkpoint_path,
                monitor=monitor,
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
        ]
        ft_history = model.fit(
            train_dataset,
            epochs=fine_tune_epochs,
            validation_data=val_dataset,
            callbacks=ft_callbacks,
            verbose=1,
        )
        if os.path.exists(ft_checkpoint_path):
            model.load_weights(ft_checkpoint_path)
        for key in ft_history.history:
            if key in history.history:
                history.history[key].extend(ft_history.history[key])
            else:
                pad = [float("nan")] * final_train_epochs
                history.history[key] = pad + ft_history.history[key]

    eval_results = model.evaluate(
        val_dataset,
        return_dict=True,
        verbose=0,
    )
    best_score = _score_from_eval_results(
        eval_results,
        target_standard_deviations=target_standard_deviations,
        target_keys=target_keys,
    )
    print(f"Final {monitor}: {best_score:.4f}")

    export_model = build_export_model(model)
    export_model.save(model_checkpoint_path)
    print(f"Model exported as '{model_checkpoint_path}'")

    if os.path.abspath(model_checkpoint_path) != os.path.abspath(final_model_path):
        export_model.save(final_model_path)
        print(f"Model exported as '{final_model_path}'")

    # Cleanup intermediate weight files
    for path in [weights_checkpoint_path,
                 f"{os.path.splitext(weights_checkpoint_path)[0]}_ft.weights.h5"]:
        if os.path.exists(path):
            os.remove(path)

    return {
        "model": export_model,
        "history": history,
        "best_hp": best_hp,
        "tuner": tuner,
        "loss_weights": loss_weights,
        "best_score": best_score,
        "fine_tune_start_epoch": final_train_epochs if fine_tune_epochs > 0 else None,
    }


def plot_training_history(
    history,
    *,
    figsize: tuple[float, float] = (7.1, 4.8),
    base_font: float = 11.25,
    output_png: str = "training_curves_elsevier.png",
    output_pdf: str | None = "training_curves_elsevier.pdf",
) -> None:
    """Plot paper-style training curves and mark the selected validation epoch."""
    hist = history.history if hasattr(history, "history") else dict(history)

    def get_metric(*names: str) -> np.ndarray:
        for name in names:
            if name in hist:
                return np.asarray(hist[name], dtype=float)
        raise KeyError(f"None of these keys were found: {names}")

    epochs = np.arange(1, len(get_metric("loss")) + 1)
    train_loss = get_metric("loss")
    val_loss = get_metric("val_loss")
    best_idx = int(np.argmin(val_loss))
    best_epoch = int(epochs[best_idx])

    plots = [
        ("Total Loss", "Huber loss", train_loss, val_loss),
        (
            r"$T_{\mathrm{eff}}$",
            "Scaled MAE",
            get_metric("teff_mae", "Teff_mae", "teff_output_mae"),
            get_metric("val_teff_mae", "val_Teff_mae", "val_teff_output_mae"),
        ),
        (
            r"$[\mathrm{Fe}/\mathrm{H}]$",
            "Scaled MAE",
            get_metric("feh_mae", "FeH_mae", "feh_output_mae"),
            get_metric("val_feh_mae", "val_FeH_mae", "val_feh_output_mae"),
        ),
        (
            r"$\log g$",
            "Scaled MAE",
            get_metric("logg_mae", "log_g_mae", "logg_output_mae"),
            get_metric("val_logg_mae", "val_log_g_mae", "val_logg_output_mae"),
        ),
    ]

    local_rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base_font,
        "axes.labelsize": base_font + 1,
        "axes.titlesize": base_font + 1,
        "xtick.labelsize": base_font,
        "ytick.labelsize": base_font,
        "legend.fontsize": base_font,
        "axes.linewidth": 0.8,
    }

    with plt.rc_context(local_rc):
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=300)
        axes = axes.ravel()

        for i, (ax, (title, ylabel, train, val)) in enumerate(zip(axes, plots)):
            ax.plot(epochs, train, color="black", linestyle="-", linewidth=1.1, label="Training")
            ax.plot(epochs, val, color="black", linestyle="--", linewidth=1.1, label="Validation")
            ax.plot(
                best_epoch,
                val[best_idx],
                marker="*",
                color="black",
                markersize=8,
                linestyle="None",
                label="Selected epoch" if i == 0 else None,
                zorder=5,
            )
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            if i >= 2:
                ax.set_xlabel("Epoch")
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            ax.grid(True, linestyle=":", linewidth=0.45, color="0.70")
            ax.legend(frameon=False, loc="best")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.tight_layout()
        if output_png:
            fig.savefig(output_png, dpi=300, bbox_inches="tight")
        if output_pdf:
            fig.savefig(output_pdf, bbox_inches="tight")
        plt.show()

    print(f"Selected epoch: {best_epoch}")
