"""Run test-set inference and save a publication-style prediction scatter PNG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_definitions.model_definitions import WarmupCosineDecay  # noqa: F401


DEFAULT_DATA = PROJECT_ROOT / "data" / "sdss_dr12_processed_flux_benchmark.npz"
DEFAULT_OUT = PROJECT_ROOT / "figs" / "no_gating_test_reference_vs_predicted.png"
DEFAULT_METRICS = PROJECT_ROOT / "results_and_evaluations" / "no_gating_test_metrics.json"

TARGETS = (
    {
        "key": "Teff",
        "title": r"$T_{\mathrm{eff}}$ (K)",
        "mae_fmt": ".1f",
        "unit": "K",
    },
    {
        "key": "FeH",
        "title": r"$[\mathrm{Fe}/\mathrm{H}]$ (dex)",
        "mae_fmt": ".3f",
        "unit": "dex",
    },
    {
        "key": "logg",
        "title": r"$\log g$ (dex)",
        "mae_fmt": ".3f",
        "unit": "dex",
    },
)


def _stack_predictions(predictions) -> np.ndarray:
    if isinstance(predictions, (list, tuple)):
        return np.column_stack([np.asarray(pred).reshape(-1) for pred in predictions])
    return np.asarray(predictions)


def _inverse_scaled_targets(values: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return values.astype(np.float64) * scales.reshape(1, -1) + centers.reshape(1, -1)


def _load_scaler_arrays(path: Path | None, data) -> tuple[np.ndarray, np.ndarray, str]:
    if path is None:
        return (
            np.asarray(data["label_robust_center"], dtype=np.float64),
            np.asarray(data["label_robust_scale"], dtype=np.float64),
            "compact NPZ label_robust_center/label_robust_scale",
        )
    scaler_data = np.load(path, allow_pickle=True)
    if "centers" in scaler_data.files and "scales" in scaler_data.files:
        return (
            np.asarray(scaler_data["centers"], dtype=np.float64),
            np.asarray(scaler_data["scales"], dtype=np.float64),
            str(path),
        )
    if "label_robust_center" in scaler_data.files and "label_robust_scale" in scaler_data.files:
        return (
            np.asarray(scaler_data["label_robust_center"], dtype=np.float64),
            np.asarray(scaler_data["label_robust_scale"], dtype=np.float64),
            str(path),
        )
    raise KeyError(f"Could not find centers/scales arrays in {path}")


def _log_point_density(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    density = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    return np.log10(np.maximum(density, np.finfo(float).tiny))


def run_inference_and_plot(args: argparse.Namespace) -> dict[str, object]:
    data = np.load(args.data, allow_pickle=True)
    x_test = np.asarray(data["X_test_features"], dtype=np.float32)
    y_test_scaled = np.asarray(data["y_test_targets"], dtype=np.float32)
    pred_centers, pred_scales, pred_scaler_source = _load_scaler_arrays(args.prediction_scalers, data)

    model = tf.keras.models.load_model(args.model, compile=False, safe_mode=False)
    predictions_scaled = _stack_predictions(
        model.predict(x_test, batch_size=args.batch_size, verbose=1),
    )

    if "y_test_targets_original" in data.files:
        y_ref = np.asarray(data["y_test_targets_original"], dtype=np.float64)
    else:
        ref_centers, ref_scales, _ = _load_scaler_arrays(None, data)
        y_ref = _inverse_scaled_targets(y_test_scaled, ref_centers, ref_scales)
    y_pred = _inverse_scaled_targets(predictions_scaled, pred_centers, pred_scales)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": args.base_font,
            "axes.titlesize": args.base_font + 1,
            "axes.labelsize": args.base_font,
            "xtick.labelsize": args.base_font - 1,
            "ytick.labelsize": args.base_font - 1,
            "legend.fontsize": args.base_font - 1,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=args.figsize, dpi=args.dpi, constrained_layout=True)
    metrics: dict[str, object] = {
        "model": str(args.model),
        "data": str(args.data),
        "prediction_scaler": pred_scaler_source,
        "output_png": str(args.out),
    }
    last_scatter = None

    for idx, (ax, target) in enumerate(zip(axes, TARGETS)):
        ref = y_ref[:, idx]
        pred = y_pred[:, idx]
        density = _log_point_density(ref, pred)
        order = np.argsort(density)
        ref_sorted = ref[order]
        pred_sorted = pred[order]
        density_sorted = density[order]

        last_scatter = ax.scatter(
            ref_sorted,
            pred_sorted,
            c=density_sorted,
            s=args.point_size,
            cmap="viridis",
            alpha=args.alpha,
            linewidths=0,
            rasterized=True,
        )

        lim_min = float(np.nanmin(np.concatenate([ref, pred])))
        lim_max = float(np.nanmax(np.concatenate([ref, pred])))
        pad = 0.03 * (lim_max - lim_min)
        lims = (lim_min - pad, lim_max + pad)
        ax.plot(lims, lims, color="red", linestyle="--", linewidth=1.2, alpha=0.85)
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)

        mae = float(mean_absolute_error(ref, pred))
        r2 = float(r2_score(ref, pred))
        metrics[target["key"]] = {
            "mae": mae,
            "r2": r2,
            "unit": target["unit"],
        }
        ax.text(
            0.045,
            0.955,
            rf"$\mathrm{{MAE}}={mae:{target['mae_fmt']}}$" + "\n" + rf"$R^2={r2:.3f}$",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=args.annotation_font,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.78, edgecolor="0.65"),
        )
        ax.set_title(target["title"])
        ax.set_xlabel("Reference")
        ax.set_ylabel("Predicted" if idx == 0 else "")
        ax.grid(True, linestyle="-", alpha=0.22)

    if last_scatter is not None:
        colorbar = fig.colorbar(last_scatter, ax=axes, fraction=0.025, pad=0.012)
        colorbar.set_label("Log point density")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved figure: {args.out}")
    print(f"Saved metrics: {args.metrics}")
    for target in TARGETS:
        item = metrics[target["key"]]
        print(f"{target['key']}: MAE={item['mae']:.6g} {item['unit']}, R2={item['r2']:.6g}")
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run model inference on the HF compact test set and save a scatter PNG.")
    parser.add_argument("--model", type=Path, required=True, help="Path to a saved Keras model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Compact HF benchmark NPZ.")
    parser.add_argument(
        "--prediction-scalers",
        type=Path,
        default=None,
        help="Optional scaler NPZ used to inverse-transform model outputs. Use this for legacy Colab models.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output PNG path.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="Output metrics JSON path.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--base-font", type=int, default=14)
    parser.add_argument("--annotation-font", type=int, default=13)
    parser.add_argument("--point-size", type=float, default=1.3)
    parser.add_argument("--alpha", type=float, default=0.78)
    parser.add_argument("--figsize", type=float, nargs=2, default=(8.0, 2.2))
    return parser


def main() -> None:
    run_inference_and_plot(build_argparser().parse_args())


if __name__ == "__main__":
    main()
