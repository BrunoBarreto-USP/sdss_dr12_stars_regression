"""Summarize fixed-checkpoint errors over the full retained target ranges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from generate_paper_figures import Figures, TARGETS, digest

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("Teff", "[Fe/H]", "log g")
EDGES = (
    np.array([4000, 4500, 5000, 5500, 6000, 6500, 7500, 8500, 9500]),
    np.array([-4.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.75]),
    np.array([0, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5]),
)


def summarize(true, pred, *, n_boot=10000, seed=42):
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if true.ndim != 2 or true.shape[1] != 3 or pred.shape != true.shape:
        raise ValueError("Expected matched N-by-3 physical labels and predictions.")
    if not np.isfinite(true).all() or not np.isfinite(pred).all() or n_boot < 1:
        raise ValueError(
            "Finite labels/predictions and a positive bootstrap count are required."
        )
    rng = np.random.default_rng(seed)
    rows = []
    for j, (target, edges) in enumerate(zip(LABELS, EDGES)):
        bins = np.searchsorted(edges, true[:, j], side="right") - 1
        bins[true[:, j] == edges[-1]] = len(edges) - 2
        if np.any((bins < 0) | (bins >= len(edges) - 1)):
            raise ValueError(f"Configured bins do not cover every {target} label.")
        for k, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            residuals = pred[bins == k, j] - true[bins == k, j]
            n = len(residuals)
            row = {
                "target": target,
                "low": float(low),
                "high": float(high),
                "n": n,
                "sparse": n < 30,
                "mae": None,
                "bias": None,
                "mae_ci_low": None,
                "mae_ci_high": None,
                "bias_ci_low": None,
                "bias_ci_high": None,
            }
            if n:
                row.update(
                    mae=float(abs(residuals).mean()), bias=float(residuals.mean())
                )
                # Bounded batches keep even the most populated bins inexpensive in RAM.
                draws = np.empty((n_boot, 2))
                for start in range(0, n_boot, 64):
                    stop = min(start + 64, n_boot)
                    resamples = residuals[rng.integers(0, n, size=(stop - start, n))]
                    draws[start:stop, 0] = abs(resamples).mean(axis=1)
                    draws[start:stop, 1] = resamples.mean(axis=1)
                ci = np.percentile(draws, [2.5, 97.5], axis=0)
                row.update(
                    mae_ci_low=float(ci[0, 0]),
                    mae_ci_high=float(ci[1, 0]),
                    bias_ci_low=float(ci[0, 1]),
                    bias_ci_high=float(ci[1, 1]),
                )
            rows.append(row)
        subset = rows[-(len(edges) - 1) :]
        assert sum(r["n"] for r in subset) == len(true)
        np.testing.assert_allclose(
            sum(r["n"] * r["mae"] for r in subset if r["n"]) / len(true),
            abs(pred[:, j] - true[:, j]).mean(),
            rtol=1e-12,
        )
    return pd.DataFrame(rows)


def plot(figures, frame, *, n_boot, seed):
    fig, axes = figures.axes(
        rows=3,
        cols=3,
        height=118,
        gridspec_kw={"height_ratios": [1.1, 1.1, 0.75]},
        sharex="col",
    )
    titles = (r"$T_{\mathrm{eff}}$", r"$[\mathrm{Fe/H}]$", r"$\log g$")
    for j, (target, unit, title, edges) in enumerate(
        zip(LABELS, ("K", "dex", "dex"), titles, EDGES)
    ):
        data = frame[frame.target == target]
        centers = (data.low.to_numpy() + data.high.to_numpy()) / 2
        for row_index, metric, label in (
            (0, "mae", "MAE"),
            (1, "bias", "Mean residual"),
        ):
            ax = axes[row_index, j]
            for sparse, color, fill in (
                (False, "black", "black"),
                (True, "#b2182b", "white"),
            ):
                good = (data.n.to_numpy() > 0) & (data["sparse"].to_numpy() == sparse)
                value = data[metric].to_numpy()[good]
                low = data[metric + "_ci_low"].to_numpy()[good]
                high = data[metric + "_ci_high"].to_numpy()[good]
                ax.errorbar(
                    centers[good],
                    value,
                    yerr=[np.maximum(value - low, 0), np.maximum(high - value, 0)],
                    fmt="o",
                    markersize=3,
                    color=color,
                    markerfacecolor=fill,
                    capsize=2,
                    linewidth=0.7,
                )
            ax.set_ylabel(f"{label} ({unit})", fontsize=8)
            ax.grid(True)
            ax.tick_params(labelsize=7.5)
        axes[0, j].set_title(title)
        axes[0, j].set_ylim(bottom=0)
        axes[1, j].axhline(0, color=".45", linestyle="--", linewidth=0.6)
        count = axes[2, j]
        count.bar(
            centers,
            data.n,
            width=np.diff(edges) * 0.85,
            color=".55",
            edgecolor="black",
            linewidth=0.4,
        )
        count.set_yscale("log")
        count.set_ylim(10, 15000)
        count.set_ylabel("Test objects", fontsize=8)
        count.set_xlabel(f"Reference {title} ({unit})", fontsize=8)
        count.tick_params(labelsize=7.5)
        count.set_xlim(edges[0], edges[-1])
        count.set_xticks(
            ([4000, 6000, 8000, 9500], [-4, -3, -2, -1, 0], [0, 1, 2, 3, 4, 5])[j]
        )
        count.grid(True, axis="y")
    figures.save(
        "parameter_range_errors_single_column",
        fig,
        n_test=int(frame[frame.target == LABELS[0]].n.sum()),
        n_boot=n_boot,
        seed=seed,
        residual="prediction minus SSPP reference",
        sparse_bin_threshold=30,
        bin_edges={t: e.tolist() for t, e in zip(LABELS, EDGES)},
        intervals="individual percentile bootstrap, conditional on the fixed checkpoint",
    )


def analyze(figures, *, n_boot=10000, seed=42):
    path = figures.run / "test_predictions.npz"
    with np.load(path, allow_pickle=False) as z:
        ids, true, pred = z["test_ids"], z["y_true"], z["y_pred"]
        if tuple(z["target_names"]) != LABELS or len(np.unique(ids)) != len(ids):
            raise ValueError("Invalid target order or repeated test IDs.")
    metadata = (
        pq.read_table(figures.root / "data/test.parquet", columns=["spec_id", *TARGETS])
        .to_pandas()
        .set_index("spec_id")
    )
    np.testing.assert_allclose(
        metadata.loc[ids, list(TARGETS)].to_numpy(), true, atol=0.001, rtol=1e-6
    )
    frame = summarize(true, pred, n_boot=n_boot, seed=seed)
    figures.run.mkdir(parents=True, exist_ok=True)
    frame.to_csv(figures.run / "parameter_range_errors.csv", index=False)
    report = {
        "predictions_sha256": digest(path),
        "n_test": len(ids),
        "n_boot": n_boot,
        "seed": seed,
        "bias_definition": "prediction minus SSPP reference",
        "bin_policy": "left-closed, right-open; final interval includes right edge",
        "sparse_bin_threshold": 30,
        "rows": json.loads(frame.to_json(orient="records", double_precision=15)),
    }
    (figures.run / "parameter_range_errors.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    plot(figures, frame, n_boot=n_boot, seed=seed)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "outputs/paper_figures/single_column/corrected_best_trial",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs/paper_figures/single_column"
    )
    parser.add_argument("--paper-dir", type=Path)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    figures = Figures(ROOT, args.out_dir, args.paper_dir, args.run_dir)
    figures.manifest = json.loads(
        (args.out_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if figures.manifest["evaluated_model"]["sha256"] != digest(
        args.run_dir / "retrained_paper_model.keras"
    ):
        raise ValueError(
            "The parameter diagnostics and article must use the same model."
        )
    frame = analyze(figures, n_boot=args.n_boot, seed=args.seed)
    figures.finish()
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
