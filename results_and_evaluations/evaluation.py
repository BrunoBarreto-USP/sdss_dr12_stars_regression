"""Evaluation, plotting, and export helpers."""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from model_definitions import model_definitions as _model_definitions  # noqa: F401
except ImportError:
    from model_definitions import model_definitions as _model_definitions  # noqa: F401


def configure_plot_style(base_font: int = 25) -> None:
    plt.rcParams.update(
        {
            "font.size": base_font,
            "axes.titlesize": base_font + 4,
            "axes.labelsize": base_font + 2,
            "xtick.labelsize": base_font,
            "ytick.labelsize": base_font,
            "legend.fontsize": base_font,
            "figure.titlesize": base_font + 4,
            "mathtext.fontset": "stix",
            "font.family": "serif",
        }
    )


def pretty_target_names(target_names: Sequence[str]) -> list[str]:
    names = []
    for n in target_names:
        n_low = str(n).lower()
        if n_low in ["teff", "t_eff", "t eff", "effective temperature"]:
            names.append(r"$T_{\mathrm{eff}}$ (K)")
        elif "fe" in n_low:
            names.append(r"$[\mathrm{Fe}/\mathrm{H}]$")
        elif "log" in n_low and "g" in n_low:
            names.append(r"$\log g$")
        else:
            names.append(str(n))
    return names


def _extract_mu(raw_preds: list[np.ndarray]) -> list[np.ndarray]:
    """Extract the mean (mu) from each task's (mu, log_var) output."""
    mu_list = []
    for p in raw_preds:
        if p.ndim > 1 and p.shape[-1] == 2:
            mu_list.append(p[:, 0:1])
        else:
            mu_list.append(p)
    return mu_list


def _extract_sigma(raw_preds: list[np.ndarray]) -> np.ndarray | None:
    """Extract per-sample predicted std from each task's (mu, raw_var) output.

    Matches the historical softplus variance parameterization:
    var = softplus(raw) + 1e-6, sigma = sqrt(var).
    """
    sigma_cols = []
    for p in raw_preds:
        if p.ndim > 1 and p.shape[-1] == 2:
            var = np.log1p(np.exp(p[:, 1:2])) + 1e-6  # softplus
            sigma_cols.append(np.sqrt(var))
    if not sigma_cols:
        return None
    return np.column_stack(sigma_cols)


def load_model_and_predict(
    model_path: str,
    X_test_features: np.ndarray,
    *,
    batch_size: int = 1024,
) -> tuple[tf.keras.Model, list[np.ndarray], np.ndarray]:
    best_model = tf.keras.models.load_model(model_path, compile=False)
    raw_preds = best_model.predict(X_test_features, batch_size=batch_size, verbose=0)
    if not isinstance(raw_preds, list):
        raw_preds = [raw_preds]
    mu_list = _extract_mu(raw_preds)
    predictions_scaled = np.column_stack(mu_list)
    return best_model, mu_list, predictions_scaled


def mc_dropout_predict(
    model: tf.keras.Model,
    X: np.ndarray,
    *,
    n_passes: int = 50,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo Dropout inference: keep dropout active over n_passes forward passes.

    Returns
    -------
    mean : (N, n_outputs) – predictive mean (use as point estimate)
    std  : (N, n_outputs) – predictive std  (per-sample epistemic uncertainty)
    """
    n_samples = X.shape[0]
    all_passes = []
    for _ in range(n_passes):
        pass_preds = []
        for start in range(0, n_samples, batch_size):
            batch = X[start : start + batch_size]
            out = model(batch, training=True)  # dropout stays ON
            if isinstance(out, (list, tuple)):
                mu_list = _extract_mu([np.asarray(p) for p in out])
                out = np.column_stack(mu_list)
            else:
                out = np.asarray(out)
                if out.ndim > 1 and out.shape[-1] == 2:
                    out = out[:, 0:1]
            pass_preds.append(out)
        all_passes.append(np.concatenate(pass_preds, axis=0))
    stacked = np.stack(all_passes, axis=0)  # (n_passes, N, n_outputs)
    return stacked.mean(axis=0), stacked.std(axis=0)


def print_test_metrics(
    predictions_scaled: np.ndarray,
    y_test_targets: np.ndarray,
    target_scalers: Sequence[object],
    target_names: Sequence[str],
) -> None:
    pretty_names = pretty_target_names(target_names)
    print("--- Final model performance on the test set ---")
    for idx, name in enumerate(pretty_names):
        pred_scaled = predictions_scaled[:, idx].reshape(-1, 1)
        pred_unscaled = target_scalers[idx].inverse_transform(pred_scaled).ravel()

        true_scaled = y_test_targets[:, idx].reshape(-1, 1)
        true_unscaled = target_scalers[idx].inverse_transform(true_scaled).ravel()

        mae = mean_absolute_error(true_unscaled, pred_unscaled)
        r2 = r2_score(true_unscaled, pred_unscaled)
        print(f"{name}: MAE={mae:.3g}, R^2={r2:.3g}")


def plot_true_vs_predicted(
    predictions_scaled_list: Sequence[np.ndarray],
    y_test_targets: np.ndarray,
    target_scalers: Sequence[object],
    target_names: Sequence[str],
    base_font: int = 25,
) -> None:
    pretty_names = pretty_target_names(target_names)
    n_params = len(pretty_names)

    fig, axes = plt.subplots(n_params, 1, figsize=(7.2, 5.4 * n_params), dpi=300)
    if n_params == 1:
        axes = [axes]

    for idx, name in enumerate(pretty_names):
        ax = axes[idx]
        pred_unscaled = target_scalers[idx].inverse_transform(predictions_scaled_list[idx].reshape(-1, 1)).ravel()
        true_unscaled = target_scalers[idx].inverse_transform(y_test_targets[:, idx].reshape(-1, 1)).ravel()

        xy = np.vstack([true_unscaled, pred_unscaled])
        density = gaussian_kde(xy)(xy)
        order = density.argsort()
        x = true_unscaled[order]
        y = pred_unscaled[order]
        d = density[order]

        ax.scatter(x, y, c=d, s=10, cmap="viridis", alpha=0.75, linewidths=0)

        lim_min = np.min(np.concatenate([x, y]))
        lim_max = np.max(np.concatenate([x, y]))
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", alpha=0.85)

        mae = mean_absolute_error(true_unscaled, pred_unscaled)
        r2 = r2_score(true_unscaled, pred_unscaled)
        ax.text(
            0.04,
            0.96,
            rf"$\mathrm{{MAE}}={mae:.3g}$" + "\n" + rf"$R^2={r2:.3g}$",
            transform=ax.transAxes,
            va="top",
            fontsize=30,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.70, edgecolor="0.3"),
        )
        ax.tick_params(axis='both', labelsize=30)
        ax.set_title(name, fontsize = 30)
        if idx == n_params - 1:
            ax.set_xlabel("True", fontsize = 30)
        else:
            ax.set_xlabel("")
        ax.set_ylabel("Predicted", fontsize = 30)
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout(h_pad=1.0)
    plt.show()


def print_parameter_ranges(
    y_train_targets: np.ndarray,
    y_test_targets: np.ndarray,
    target_scalers: Sequence[object],
    target_cols: Sequence[int],
    target_names: Sequence[str],
) -> None:
    print("train params ranges")
    for idx, _ in enumerate(target_names):
        col_idx = target_cols[idx]
        col_name = target_names[idx]
        min_val = target_scalers[idx].inverse_transform([[y_train_targets[:, idx].min()]])[0, 0]
        max_val = target_scalers[idx].inverse_transform([[y_train_targets[:, idx].max()]])[0, 0]
        print(f"{col_name} (col {col_idx}): {min_val:.3f} to {max_val:.3f}")

    print("test params ranges")
    for idx, _ in enumerate(target_names):
        col_idx = target_cols[idx]
        col_name = target_names[idx]
        min_val = target_scalers[idx].inverse_transform([[y_test_targets[:, idx].min()]])[0, 0]
        max_val = target_scalers[idx].inverse_transform([[y_test_targets[:, idx].max()]])[0, 0]
        print(f"{col_name} (col {col_idx}): {min_val:.3f} to {max_val:.3f}")


def export_predictions_and_truth(
    data_dir: str,
    test_ids: np.ndarray,
    predictions_scaled: np.ndarray,
    y_test_targets: np.ndarray,
    target_scalers: Sequence[object],
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    predictions_unscaled = np.column_stack(
        [
            target_scalers[idx].inverse_transform(predictions_scaled[:, [idx]]).ravel()
            for idx in range(predictions_scaled.shape[1])
        ]
    )
    y_test_unscaled = np.column_stack(
        [
            target_scalers[idx].inverse_transform(y_test_targets[:, [idx]]).ravel()
            for idx in range(y_test_targets.shape[1])
        ]
    )

    pred_df = pd.DataFrame(
        {
            "ID": test_ids,
            "Teff": predictions_unscaled[:, 0],
            "FeH": predictions_unscaled[:, 1],
            "logg": predictions_unscaled[:, 2],
        }
    )
    ground_truth_df = pd.DataFrame(
        {
            "ID": test_ids,
            "Teff": y_test_unscaled[:, 0],
            "FeH": y_test_unscaled[:, 1],
            "logg": y_test_unscaled[:, 2],
        }
    )

    pred_path = os.path.join(data_dir, "stellar_regression_model_predictions.csv")
    truth_path = os.path.join(data_dir, "stellar_regression_model_test_groundtruth.csv")

    pred_df.to_csv(pred_path, index=False)
    ground_truth_df.to_csv(truth_path, index=False)
    print("Saved CSV files:")
    print(f"  Predictions -> {pred_path}")
    print(f"  Ground truth -> {truth_path}")
    return pred_df, ground_truth_df, pred_path, truth_path


def plot_gate_analysis(
    model: tf.keras.Model,
    X: np.ndarray,
    *,
    task_labels: Sequence[str] = (r"$T_{\mathrm{eff}}$", r"$[\mathrm{Fe/H}]$", r"$\log\,g$"),
    threshold_off: float = 0.1,
    threshold_on: float = 0.9,
    base_font: int = 13,
) -> dict[str, np.ndarray]:
    """Visualise per-task sigmoid gate activations.

    Produces three panels:
    1. **Mean gate activation per unit** – sorted bar chart for each task.
    2. **Sparsity summary** – stacked bar showing the fraction of units that are
       effectively off (<threshold_off), intermediate, or on (>threshold_on).
    3. **Task overlap heatmap** – cosine similarity between average gate vectors.

    Returns a dict mapping ``"task_0_gate"`` etc. to the ``(N, shared_dim)``
    activation arrays for further analysis.
    """
    base = getattr(model, "base_model", model)

    # Discover gate layers
    gate_layers = sorted(
        [l for l in base.layers if l.name.endswith("_gate")],
        key=lambda l: l.name,
    )
    if not gate_layers:
        raise ValueError("No gate layers found in the model.")

    # Build a sub-model that outputs gate activations
    gate_outputs = [l.output for l in gate_layers]
    gate_model = tf.keras.Model(inputs=base.inputs, outputs=gate_outputs)
    raw = gate_model.predict(X, verbose=0)
    if not isinstance(raw, list):
        raw = [raw]
    activations = {l.name: np.asarray(a) for l, a in zip(gate_layers, raw)}

    n_tasks = len(gate_layers)
    labels = list(task_labels[:n_tasks])

    # Compute mean gate vector per task  (shared_dim,)
    means = [activations[l.name].mean(axis=0) for l in gate_layers]

    local_rc = {
        "font.size": base_font,
        "axes.titlesize": base_font + 1,
        "axes.labelsize": base_font,
        "xtick.labelsize": base_font - 1,
        "ytick.labelsize": base_font - 1,
        "legend.fontsize": base_font - 1,
    }

    with plt.rc_context(local_rc):
        fig, axes_arr = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

        # --- Panel 1: sorted mean activations per task ---
        ax = axes_arr[0]
        for i, (m, lab) in enumerate(zip(means, labels)):
            ax.plot(np.sort(m)[::-1], label=lab, alpha=0.85)
        ax.axhline(threshold_off, ls="--", color="grey", lw=0.8, label=f"off < {threshold_off}")
        ax.axhline(threshold_on, ls="--", color="grey", lw=0.8)
        ax.set_xlabel("Shared unit (sorted)")
        ax.set_ylabel("Mean gate activation")
        ax.set_title("Gate activation profile")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Panel 2: sparsity stacked bar ---
        ax = axes_arr[1]
        frac_off, frac_mid, frac_on = [], [], []
        for m in means:
            n = len(m)
            frac_off.append((m < threshold_off).sum() / n)
            frac_on.append((m > threshold_on).sum() / n)
            frac_mid.append(1 - frac_off[-1] - frac_on[-1])
        x_pos = np.arange(n_tasks)
        ax.bar(x_pos, frac_off, label=f"Off (< {threshold_off})", color="#d62728")
        ax.bar(x_pos, frac_mid, bottom=frac_off, label="Intermediate", color="#ff7f0e")
        ax.bar(x_pos, frac_on, bottom=[a + b for a, b in zip(frac_off, frac_mid)],
               label=f"On (> {threshold_on})", color="#2ca02c")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Fraction of shared units")
        ax.set_title("Gate sparsity breakdown")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)

        # --- Panel 3: task overlap (cosine similarity) ---
        ax = axes_arr[2]
        mean_mat = np.stack(means)  # (n_tasks, shared_dim)
        norms = np.linalg.norm(mean_mat, axis=1, keepdims=True) + 1e-8
        cos_sim = (mean_mat @ mean_mat.T) / (norms @ norms.T)
        im = ax.imshow(cos_sim, vmin=0, vmax=1, cmap="YlOrRd")
        ax.set_xticks(range(n_tasks))
        ax.set_yticks(range(n_tasks))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for r in range(n_tasks):
            for c in range(n_tasks):
                ax.text(c, r, f"{cos_sim[r, c]:.2f}", ha="center", va="center",
                        fontsize=base_font, color="white" if cos_sim[r, c] > 0.6 else "black")
        ax.set_title("Gate overlap (cosine similarity)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.show()

        # --- Gate mask heatmap: 3 rows × shared_dim columns ---
        shared_dim = len(means[0])
        mean_mat = np.stack(means)  # (n_tasks, shared_dim)
        # Sort columns by max activation across tasks so structure is visible
        sort_idx = np.argsort(-mean_mat.max(axis=0))
        mean_sorted = mean_mat[:, sort_idx]

        aspect = max(shared_dim / (n_tasks * 12), 1.0)
        fig_h, _ax = plt.subplots(
            figsize=(min(shared_dim * 0.12, 22), 2.0),
            constrained_layout=True,
        )
        from matplotlib.colors import LinearSegmentedColormap
        black_red = LinearSegmentedColormap.from_list("black_red", ["black", "red"])
        im2 = _ax.imshow(mean_sorted, aspect=aspect, cmap=black_red, vmin=0, vmax=1)
        _ax.set_yticks(range(n_tasks))
        _ax.set_yticklabels(labels, fontsize=base_font)
        _ax.set_xlabel("Shared unit (sorted by max activation)", fontsize=base_font)
        _ax.set_title("Gate mask per task", fontsize=base_font + 1)
        _ax.tick_params(axis="x", labelsize=base_font - 2)
        fig_h.colorbar(im2, ax=_ax, fraction=0.015, pad=0.02, label="Gate activation")
        plt.show()

    # Print summary
    for i, (m, lab) in enumerate(zip(means, labels)):
        n = len(m)
        off = (m < threshold_off).sum()
        on = (m > threshold_on).sum()
        print(f"{lab}: {off}/{n} units off ({off/n:.0%}), "
              f"{on}/{n} units on ({on/n:.0%}), "
              f"{n - off - on}/{n} intermediate ({(n - off - on)/n:.0%})")

    return activations


def print_residual_statistics(
    predictions_scaled: np.ndarray,
    y_test_targets: np.ndarray,
    target_scalers: Sequence[object],
    target_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Print and return bias (mean residual) and scatter (std residual) per target.

    Residuals are computed in the original (unscaled) space as pred - true.
    """
    pretty_names = pretty_target_names(target_names)
    stats: dict[str, dict[str, float]] = {}
    print("--- Residual statistics (pred - true, original scale) ---")
    for idx, name in enumerate(pretty_names):
        pred_unscaled = target_scalers[idx].inverse_transform(
            predictions_scaled[:, idx].reshape(-1, 1)
        ).ravel()
        true_unscaled = target_scalers[idx].inverse_transform(
            y_test_targets[:, idx].reshape(-1, 1)
        ).ravel()
        residuals = pred_unscaled - true_unscaled
        bias = float(np.mean(residuals))
        scatter = float(np.std(residuals))
        stats[target_names[idx]] = {"bias": bias, "scatter": scatter}
        print(f"  {name}: bias = {bias:+.4g}, scatter (sigma) = {scatter:.4g}")
    return stats


def report_mae(true_values: np.ndarray, pred_values: np.ndarray, label: str) -> float:
    mae_value = mean_absolute_error(true_values, pred_values)
    print(f"MAE {label}: {mae_value:.4f}")
    return mae_value


def print_macro_mae(pred_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> float:
    mae_teff = report_mae(ground_truth_df["Teff"], pred_df["Teff"], "Teff")
    mae_feh = report_mae(ground_truth_df["FeH"], pred_df["FeH"], "FeH")
    mae_logg = report_mae(ground_truth_df["logg"], pred_df["logg"], "logg")
    normalized_macro_mae = (mae_teff / 100 + mae_feh / 0.20 + mae_logg / 0.20) / 3.0
    print(f"Macro MAE: {normalized_macro_mae:.4f}")
    return normalized_macro_mae


def _plot_labels_by_language(language: str) -> dict[str, object]:
    lang = language.lower()
    if lang == "en":
        return {
            "pretty_names": [
                "Teff (K)",
                "[Fe/H]",
                "log g",
            ],
            "xlabel": "True",
            "ylabel": "Predicted",
            "colorbar": "Point density",
        }
    if lang == "pt":
        return {
            "pretty_names": [
                "Teff (K)",
                "[Fe/H]",
                "log g",
            ],
            "xlabel": "Real",
            "ylabel": "Predito",
            "colorbar": "Densidade de pontos",
        }
    raise ValueError("Unsupported language. Use 'en' or 'pt'.")


def plot_from_csv(pred_path: str, true_path: str, base_font: int = 25, language: str = "en") -> None:
    pred_df = pd.read_csv(pred_path)
    true_df = pd.read_csv(true_path)
    merged = pred_df.merge(true_df, on="ID", suffixes=("_pred", "_true"))

    labels = _plot_labels_by_language(language)
    pretty_names = labels["pretty_names"]
    cols = ["Teff", "FeH", "logg"]
    n_params = len(cols)

    fig, axes = plt.subplots(1, n_params, figsize=(6.2 * n_params, 5.4), dpi=300)
    if n_params == 1:
        axes = [axes]

    last_scatter = None
    for idx, (col, name) in enumerate(zip(cols, pretty_names)):
        ax = axes[idx]
        x = merged[f"{col}_true"].values
        y = merged[f"{col}_pred"].values

        xy = np.vstack([x, y])
        density = gaussian_kde(xy)(xy)
        order = density.argsort()
        x_sorted = x[order]
        y_sorted = y[order]
        d_sorted = density[order]

        last_scatter = ax.scatter(x_sorted, y_sorted, c=d_sorted, s=10, cmap="viridis", alpha=0.75, linewidths=0)

        lim_min = np.min(np.concatenate([x, y]))
        lim_max = np.max(np.concatenate([x, y]))
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", alpha=0.85, label="_nolegend_")
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)

        mae = mean_absolute_error(x, y)
        r2 = r2_score(x, y)
        ax.text(
            0.04,
            0.96,
            rf"$\mathrm{{MAE}}={mae:.3g}$" + "\n" + rf"$R^2={r2:.3g}$",
            transform=ax.transAxes,
            va="top",
            fontsize=base_font + 1,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.70, edgecolor="0.3"),
        )
        ax.set_title(name)
        ax.set_xlabel(labels["xlabel"])
        ax.set_ylabel(labels["ylabel"])
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout(w_pad=2.6)
    cbar = fig.colorbar(last_scatter, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label(labels["colorbar"])
    plt.show()
