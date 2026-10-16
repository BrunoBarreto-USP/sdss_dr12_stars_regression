"""Generate publication figures from the local, object-deduplicated SDSS experiment."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker, colors
from matplotlib.patches import Rectangle, Circle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
TARGETS = ("catalog_teff", "catalog_feh", "catalog_logg")
TITLES = (r"$T_{\mathrm{eff}}$ (K)", r"$[\mathrm{Fe/H}]$ (dex)", r"$\log g$ (dex)")
WIDTH = 166 / 25.4
COMPACT_WIDTH_MM = 150
LOGLAM = np.linspace(3.5754, 3.9670, 4000)
WAVELENGTH = 10**LOGLAM


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def configure_style():
    font = font_manager.findfont(
        font_manager.FontProperties(family="STIXGeneral"), fallback_to_default=False
    )
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8,
            "axes.linewidth": 0.65,
            "lines.linewidth": 0.75,
            "grid.color": "0.90",
            "grid.linewidth": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "text.usetex": False,
            "figure.constrained_layout.w_pad": 0.09,
            "figure.constrained_layout.h_pad": 0.09,
        }
    )
    return font


class Figures:
    def __init__(self, root, out, paper, run_dir=None):
        self.root, self.out, self.paper = root, out, paper
        self.run = run_dir or out / "corrected_best_trial"
        out.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "project_root": str(root),
            "width_mm": 166,
            "font": configure_style(),
            "figures": {},
            "sources": {},
            "notes": [],
        }

    def save(self, name, fig, **details):
        fig.canvas.draw()
        for suffix in ("pdf", "png"):
            fig.savefig(self.out / f"{name}.{suffix}", dpi=600)
        self.manifest["figures"][name] = {
            "width_mm": round(fig.get_figwidth() * 25.4, 2),
            "height_mm": round(fig.get_figheight() * 25.4, 2),
            **details,
        }
        plt.close(fig)
        print("Saved", name, flush=True)

    def axes(self, rows=1, cols=1, height=65, width=166, **kwargs):
        return plt.subplots(
            rows,
            cols,
            figsize=(width / 25.4, height / 25.4),
            layout="constrained",
            **kwargs,
        )

    def load_metadata(self):
        columns = [
            "spec_id",
            "plate",
            "mjd",
            "fiber",
            "ra",
            "dec",
            "snr",
            "rv_adop",
            *TARGETS,
        ]
        self.frames = {
            s: pq.read_table(
                self.root / "data" / f"{s}.parquet", columns=columns
            ).to_pandas()
            for s in SPLITS
        }
        counts = {s: len(f) for s, f in self.frames.items()}
        assert counts == dict(zip(SPLITS, (29422, 4846, 14443))), counts
        self.all = pd.concat(self.frames.values(), ignore_index=True)
        assert self.all.spec_id.nunique() == len(self.all), (
            "Overlapping spectrum identifiers"
        )
        self.manifest["sample_counts"] = counts
        self.manifest["metadata_summary"] = {
            k: {
                "min": float(self.all[k].min()),
                "median": float(self.all[k].median()),
                "max": float(self.all[k].max()),
            }
            for k in ("snr", "rv_adop")
        }

    def sky(self):
        ra = np.asarray(self.all.ra)
        dec = np.asarray(self.all.dec)
        good = np.isfinite(ra) & np.isfinite(dec) & (dec >= -90) & (dec <= 90)
        ar, dr = np.deg2rad(ra[good]), np.deg2rad(dec[good])
        unit = np.column_stack(
            (np.cos(dr) * np.cos(ar), np.cos(dr) * np.sin(ar), np.sin(dr))
        )
        distance = cKDTree(unit).query(unit, k=33, workers=2)[0][:, -1]
        # The spherical-cap solid angle equals pi times squared chord distance.
        density = 32 / (np.pi * np.maximum(distance, 1e-12) ** 2)
        relative = density / density.max()
        longitude = -((ar + np.pi) % (2 * np.pi) - np.pi)
        order = np.argsort(relative)
        fig = plt.figure(figsize=(WIDTH, 83 / 25.4), layout="constrained")
        ax = fig.add_subplot(projection="aitoff")
        norm = colors.LogNorm(vmin=float(relative.min()), vmax=1)
        point_alpha = 0.35
        ax.scatter(
            longitude[order],
            dr[order],
            c=relative[order],
            s=1.1,
            cmap="viridis",
            norm=norm,
            alpha=point_alpha,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xticks(np.deg2rad([-120, -60, 0, 60, 120]))
        ax.set_xticklabels(
            [
                r"$120^\circ$",
                r"$60^\circ$",
                r"$0^\circ$",
                r"$300^\circ$",
                r"$240^\circ$",
            ]
        )
        ax.set_yticks(np.deg2rad([-60, -30, 0, 30, 60]))
        ax.grid(True)
        ax.set_xlabel("Right ascension (RA)")
        ax.set_ylabel("Declination (DEC)", labelpad=5)
        # Keep the quantitative color key opaque while overlapping points blend.
        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"),
            ax=ax,
            orientation="horizontal",
            fraction=0.055,
            pad=0.10,
            label="Relative angular sampling density (log scale)",
            aspect=35,
        )
        self.save(
            "sky_distribution_single_column",
            fig,
            n=int(good.sum()),
            point_alpha=point_alpha,
            density="32-nearest-neighbour count / spherical-cap solid angle, normalized by maximum; RA increases to the left",
        )

    def histogram(self, ax, values, bins, xlabel, density=False):
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            color="black",
            linewidth=0.8,
            density=density,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density" if density else "Number of spectra")
        ax.grid(True, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(direction="in", top=True, right=True)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

    def observations(self):
        entries = [
            (
                "snr_distribution_single_column",
                np.asarray(self.all.snr),
                np.arange(0, 141, 2),
                "Signal-to-noise ratio",
            ),
            (
                "radial_velocity_single_column",
                np.asarray(self.all.rv_adop),
                np.linspace(-1000, 1000, 101),
                r"$RV_{\mathrm{ADOP}}$ (km s$^{-1}$)",
            ),
        ]
        for name, values, bins, label in entries:
            good = np.isfinite(values) & (values != -9999)
            fig, ax = self.axes(height=62)
            self.histogram(ax, values[good], bins, label)
            self.save(name, fig, n=int(good.sum()))
        fig, axes = self.axes(cols=2, height=63)
        for i, (ax, (_, values, bins, label)) in enumerate(zip(axes, entries)):
            self.histogram(ax, values, bins, label)
            ax.set_title(f"({chr(97 + i)})", loc="left")
        self.save("observational_quality_single_column", fig, n=len(self.all))

    def distributions(self):
        fig, axes = self.axes(cols=3, height=63)
        for j, (ax, column, title) in enumerate(zip(axes, TARGETS, TITLES)):
            lo, hi = self.all[column].min(), self.all[column].max()
            bins = np.linspace(lo, hi, 61)
            for split, color, label in zip(
                SPLITS,
                ("tab:blue", "tab:orange", "tab:green"),
                ("Train", "Validation", "Test"),
            ):
                ax.hist(
                    self.frames[split][column],
                    bins=bins,
                    histtype="step",
                    density=True,
                    linewidth=0.65,
                    color=color,
                    label=label,
                )
            ax.set_xlabel(title)
            ax.set_title(f"({chr(97 + j)})", loc="left")
            ax.set_ylabel("Density" if j == 0 else "")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            if j == 0:
                ax.ticklabel_format(
                    axis="y", style="sci", scilimits=(-3, 3), useMathText=True
                )
            ax.grid(True)
        axes[0].legend(frameon=False, fontsize=7.7, loc="upper right")
        self.save(
            "target_distributions_single_column",
            fig,
            counts=self.manifest["sample_counts"],
        )

    def raw_diagnostics(self):
        cache = self.out / "raw_flux_diagnostics.npz"
        if cache.exists():
            z = np.load(cache)
            means = z["mean_flux"]
            selected = {k: z[k] for k in z.files if k != "mean_flux"}
        else:
            means = []
            minimum = np.inf
            maximum = -np.inf
            selected = None
            for split in SPLITS:
                offset = 0
                for batch in pq.ParquetFile(
                    self.root / "data" / f"{split}.parquet"
                ).iter_batches(
                    batch_size=256,
                    columns=["spec_id", "flux", "loglam", "processed_flux", "rv_adop"],
                    use_threads=False,
                ):
                    frame = batch.to_pandas()
                    for local, row in frame.iterrows():
                        flux = np.asarray(row.flux, dtype=float)
                        loglam = np.asarray(row.loglam, dtype=float)
                        valid = np.isfinite(flux)
                        means.append(
                            float(np.mean(flux[valid])) if valid.any() else np.nan
                        )
                        minimum = min(minimum, float(np.min(loglam)))
                        maximum = max(maximum, float(np.max(loglam)))
                        if split == "train" and offset + local == 123:
                            selected = {
                                "raw_flux": flux,
                                "raw_loglam": loglam,
                                "processed_flux": np.asarray(row.processed_flux),
                                "spec_id": np.asarray(row.spec_id),
                                "rv_adop": np.asarray(row.rv_adop),
                            }
                    offset += len(frame)
                print("Read raw spectra:", split, offset, flush=True)
            means = np.asarray(means)
            selected.update(
                {
                    "observed_loglam_min": np.asarray(minimum),
                    "observed_loglam_max": np.asarray(maximum),
                }
            )
            np.savez_compressed(cache, mean_flux=means, **selected)
        good = np.isfinite(means) & (means > 0)
        fig, ax = self.axes(height=61, width=COMPACT_WIDTH_MM)
        self.histogram(
            ax, np.log10(means[good]), 70, r"$\log_{10}(\overline{F})$", density=True
        )
        self.save(
            "flux_scale_single_column",
            fig,
            n=int(good.sum()),
            excluded=int((~good).sum()),
            stage="Mean finite native observed-frame flux before interpolation and standardization",
        )
        fig, axes = self.axes(cols=2, height=67)
        axes[0].plot(
            10 ** selected["raw_loglam"],
            selected["raw_flux"],
            color="black",
            linewidth=0.4,
        )
        axes[1].plot(
            WAVELENGTH, selected["processed_flux"], color="black", linewidth=0.4
        )
        for i, ax in enumerate(axes):
            ax.set_xlabel(r"Wavelength ($\AA$)")
            ax.set_title(
                ("(a) Observed spectrum", "(b) Preprocessed spectrum")[i], loc="left"
            )
            ax.set_xlim(3762, 9268)
            ax.set_xticks([4000, 6000, 8000])
            ax.grid(True)
        axes[0].set_ylabel(r"Flux ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")
        axes[1].set_ylabel("Standardized flux")
        self.save(
            "spectrum_preprocessing_single_column",
            fig,
            spec_id=int(selected["spec_id"]),
            wavelength_grid={"log10_start": 3.5754, "log10_end": 3.967, "points": 4000},
            native_loglam_extrema=[
                float(selected["observed_loglam_min"]),
                float(selected["observed_loglam_max"]),
            ],
        )

    def optimization(self):
        folder = (
            self.root
            / "results_and_evaluations/keras_tuner/stellar_baseline_orchestration"
        )
        trials = [
            json.loads(p.read_text()) for p in sorted(folder.glob("trial_*/trial.json"))
        ]
        complete = [
            t for t in trials if t["status"] == "COMPLETED" and t["score"] is not None
        ]
        scores = np.asarray([t["score"] for t in complete])
        numbers = np.arange(1, len(scores) + 1)
        best = np.minimum.accumulate(scores)
        fig, ax = self.axes(height=60, width=COMPACT_WIDTH_MM)
        ax.plot(
            numbers,
            scores,
            "o-",
            markersize=2.2,
            markerfacecolor="white",
            markeredgewidth=0.5,
            color="black",
            linewidth=0.55,
            label="Trial score",
        )
        ax.plot(numbers, best, "--", color="black", linewidth=0.9, label="Best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Normalized validation MAE score")
        ax.set_xlim(1, len(scores))
        ax.grid(True)
        ax.legend(frameon=False)
        winning = complete[int(np.argmin(scores))]
        self.save(
            "bayesian_optimization_single_column",
            fig,
            n_trials=len(scores),
            best_trial_id=winning["trial_id"],
            best_score=float(scores.min()),
            best_hyperparameters=winning["hyperparameters"]["values"],
        )
        pd.DataFrame({"trial": numbers, "score": scores, "best_so_far": best}).to_csv(
            self.out / "optimization_scores.csv", index=False
        )

    def weights(self):
        path = self.run / "retrained_paper_model.keras"
        with zipfile.ZipFile(path) as archive:
            config = json.loads(archive.read("config.json"))
            with h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r") as h5:
                datasets = []
                h5.visititems(
                    lambda name, item: (
                        datasets.append((name, item.shape))
                        if isinstance(item, h5py.Dataset) and name.startswith("layers/")
                        else None
                    )
                )
                kernel_name = next(
                    name for name, shape in datasets if tuple(shape) == (4000, 128)
                )
                w = h5[kernel_name][()]
                count = sum(int(np.prod(shape)) for _, shape in datasets)
        layers = config["config"]["layers"]
        dense = [
            {"name": l["config"]["name"], "units": l["config"]["units"]}
            for l in layers
            if l["class_name"] == "Dense"
        ]
        self.manifest["evaluated_model"] = {
            "path": str(path),
            "sha256": digest(path),
            "trainable_parameters": count,
            "dense_layers": dense,
        }
        assert count == 530275, count
        importance = np.linalg.norm(w, axis=1)
        importance /= importance.max()
        fig = plt.figure(figsize=(WIDTH, 101 / 25.4), layout="constrained")
        gs = fig.add_gridspec(2, 1, height_ratios=[0.80, 1], hspace=0.07)
        upper_grid = gs[0].subgridspec(1, 2, width_ratios=[1, 0.022], wspace=0.025)
        lower_grid = gs[1].subgridspec(1, 2, width_ratios=[1.5, 1], wspace=0.12)
        ax = fig.add_subplot(upper_grid[0, 0])
        cax = fig.add_subplot(upper_grid[0, 1])
        lower = fig.add_subplot(lower_grid[0, 0])
        zoom = fig.add_subplot(lower_grid[0, 1])
        limit = float(np.max(abs(w)))
        # A dark neutral centre distinguishes near-zero coefficients from both
        # signs. Preserve symmetric limits and every original weight value.
        linear_threshold_quantile = 0.25
        linear_threshold = float(np.quantile(abs(w), linear_threshold_quantile))
        image = ax.imshow(
            w.T,
            origin="lower",
            aspect="auto",
            extent=(LOGLAM[0], LOGLAM[-1], -0.5, 127.5),
            cmap="berlin",
            norm=colors.SymLogNorm(linthresh=linear_threshold, vmin=-limit, vmax=limit),
            interpolation="nearest",
            rasterized=True,
        )
        ax.set_ylabel("Stem neuron index")
        ax.set_title("(a)", loc="left")
        ax.set_yticks([0, 32, 64, 96, 127])
        ticks = np.array([4000, 5000, 6000, 7000, 8000, 9000])
        ax.set_xticks(np.log10(ticks), labels=[str(i) for i in ticks])
        fig.colorbar(
            image,
            cax=cax,
            label="Weight",
            ticks=[-1, -0.1, 0, 0.1, 1],
        )
        zoom_range = (4800, 5300)
        lower.axvspan(
            *np.log10(zoom_range), color="#73869a", alpha=0.18, linewidth=0, zorder=0
        )
        lower.plot(LOGLAM, importance, color="black", linewidth=0.65)
        lower.set_xlim(LOGLAM[0], LOGLAM[-1])
        lower.set_ylabel("Normalized L2 norm")
        lower.set_title("(b) Full wavelength range", loc="left")
        ticks = np.array([4000, 6000, 8000])
        lower.set_xticks(np.log10(ticks), labels=[str(i) for i in ticks])
        zoom.plot(LOGLAM, importance, color="#194c70", linewidth=0.8)
        zoom.set_xlim(*np.log10(zoom_range))
        zoom.set_title(r"(c) 4800--5300 $\AA$", loc="left")
        ticks = np.array([4800, 5000, 5200])
        zoom.set_xticks(np.log10(ticks), labels=[str(i) for i in ticks])
        for panel in (lower, zoom):
            panel.set_xlabel(r"Wavelength ($\AA$)")
            panel.set_ylim(0, 1.24)
            panel.set_yticks([0, 0.5, 1])
            panel.grid(True)

        # Identify the three largest local maxima of the unmodified L2 curve.
        # No smoothing, feature attribution or physical line assignment is used.
        candidates, _ = find_peaks(importance)
        peak_indices = candidates[
            np.argsort(-importance[candidates], kind="stable")[:3]
        ]
        for index in peak_indices:
            lower.plot(
                LOGLAM[index], importance[index], "o", color="#194c70", markersize=2.5
            )
            if zoom_range[0] <= WAVELENGTH[index] <= zoom_range[1]:
                zoom.plot(
                    LOGLAM[index], importance[index], "o", color="#194c70", markersize=3
                )
                # Separate labels for the neighbouring 5174/5185-Angstrom peaks.
                if importance[index] > 0.9:
                    offset, alignment = (0, 7), "center"
                elif WAVELENGTH[index] > 5100:
                    offset, alignment = (-6, 6), "right"
                else:
                    offset, alignment = (6, 7), "left"
                zoom.annotate(
                    f"{WAVELENGTH[index]:.0f}" + r" $\AA$",
                    xy=(LOGLAM[index], importance[index]),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=7.5,
                    ha=alignment,
                    va="bottom",
                )
        self.save(
            "first_layer_weights_single_column",
            fig,
            kernel_shape=list(w.shape),
            color_normalization={
                "type": "symmetric logarithmic, no clipping",
                "linear_threshold": linear_threshold,
                "linear_threshold_abs_weight_quantile": linear_threshold_quantile,
                "max_abs": limit,
                "colormap": "berlin",
                "zero_color": "dark neutral; negative blue, positive orange-red",
            },
            wavelength_coordinate="Uniform log10(lambda) pixels with physical wavelength tick labels in all three panels",
            panels={
                "a": "Full signed weight matrix, original neuron order",
                "b": "Full unmodified normalized L2 curve; shaded interval shown in c",
                "c": "4800-5300 Angstrom zoom, same vertical scale as b",
            },
            zoom_range_angstrom=list(zoom_range),
            peak_selection="Three largest local maxima, no smoothing or minimum separation",
            annotated_peaks=[
                {
                    "pixel_index": int(i),
                    "wavelength_angstrom": float(WAVELENGTH[i]),
                    "normalized_l2": float(importance[i]),
                }
                for i in peak_indices
            ],
        )
        np.savez_compressed(
            self.out / "first_layer_weights.npz",
            kernel=w,
            wavelength=WAVELENGTH,
            normalized_l2=importance,
        )

    def predictions(self):
        path = self.run / "test_predictions.npz"
        z = np.load(path)
        ids = z["test_ids"]
        true = z["y_true"]
        pred = z["y_pred"]
        frame = self.frames["test"].set_index("spec_id").loc[ids]
        assert len(np.unique(ids)) == 14443 and len(frame) == 14443
        np.testing.assert_allclose(
            frame[list(TARGETS)].to_numpy(), true, atol=0.001, rtol=1e-6
        )
        snr = frame.snr.to_numpy()
        mae = np.mean(abs(pred - true), axis=0)
        self.manifest["test_mae"] = dict(zip(("Teff", "[Fe/H]", "log g"), mae.tolist()))
        self.manifest["test_ranges"] = np.ptp(true, axis=0).tolist()
        self.manifest["range_normalized_mae_percent"] = (
            100 * mae / np.ptp(true, axis=0)
        ).tolist()
        self.manifest["sources"]["test_predictions"] = {
            "path": str(path),
            "sha256": digest(path),
        }
        fig = plt.figure(
            figsize=(COMPACT_WIDTH_MM / 25.4, 75 / 25.4), layout="constrained"
        )
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.065])
        norm = colors.Normalize(float(snr.min()), float(snr.max()))
        point_alpha = 0.25
        point_size = 1.1
        # Draw higher-SNR spectra last so lower-SNR overlaps do not hide them.
        order = np.argsort(snr, kind="stable")
        for j in range(3):
            ax = fig.add_subplot(gs[0, j])
            ax.scatter(
                true[order, j],
                pred[order, j],
                c=snr[order],
                cmap="viridis",
                norm=norm,
                s=point_size,
                alpha=point_alpha,
                linewidths=0,
                rasterized=True,
            )
            lo = min(true[:, j].min(), pred[:, j].min())
            hi = max(true[:, j].max(), pred[:, j].max())
            pad = (hi - lo) * 0.025
            limits = (lo - pad, hi + pad)
            ax.plot(limits, limits, "--", color="firebrick", linewidth=0.7)
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Reference")
            ax.set_ylabel("Predicted" if j == 0 else "")
            ax.set_title(TITLES[j])
            ax.grid(True)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        cax = fig.add_subplot(gs[1, :])
        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"),
            cax=cax,
            orientation="horizontal",
            label="Signal-to-noise ratio",
        )
        self.save(
            "test_predictions_single_column",
            fig,
            n_test=len(ids),
            mae=mae.tolist(),
            point_alpha=point_alpha,
            point_size_pt2=point_size,
            point_order="Ascending SNR, stable; higher-SNR spectra drawn last",
            color_normalization={
                "type": "linear",
                "vmin": float(snr.min()),
                "vmax": float(snr.max()),
            },
        )
        bins = np.arange(0, 141, 20)
        statistics = []
        residual = pred - true
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (snr >= lo) & (snr < hi)
            n = int(mask.sum())
            if not n:
                continue
            row = {"snr_low": int(lo), "snr_high": int(hi), "n": n}
            for j, target in enumerate(("teff", "feh", "logg")):
                row[f"mae_{target}"] = float(np.mean(abs(residual[mask, j])))
                row[f"bias_{target}"] = float(np.mean(residual[mask, j]))
                row[f"std_{target}"] = float(np.std(residual[mask, j]))
            statistics.append(row)
        stats = pd.DataFrame(statistics)
        stats.to_csv(self.out / "snr_error_statistics.csv", index=False)
        self.manifest["snr_statistics"] = statistics

    def residual_diagram(self):
        fig, axes = self.axes(rows=2, height=78)
        for index, ax in enumerate(axes):
            ax.set_xlim(0, 10.3)
            ax.set_ylim(-0.6, 1.45)
            ax.axis("off")
            ax.set_title(
                (
                    "(a) Identity shortcut: equal widths",
                    "(b) Projected shortcut: 128 to 32 units",
                )[index],
                loc="left",
                pad=2,
            )
            xs = [1.1, 2.3, 3.5, 4.7, 5.9, 7.1]
            for x, label in zip(xs, ["LN", "Dense", "GELU", "LN", "Dense", "Dropout"]):
                ax.add_patch(
                    Rectangle(
                        (x - 0.43, -0.22),
                        0.86,
                        0.44,
                        facecolor="white",
                        edgecolor="black",
                        linewidth=0.65,
                    )
                )
                ax.text(x, 0, label, ha="center", va="center", fontsize=8)
            for left, right in zip(
                [0.2, *[x + 0.43 for x in xs[:-1]]], [*[x - 0.43 for x in xs]]
            ):
                ax.annotate(
                    "",
                    xy=(right, 0),
                    xytext=(left, 0),
                    arrowprops={"arrowstyle": "->", "lw": 0.65},
                )
            ax.add_patch(
                Circle(
                    (8.55, 0),
                    0.23,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.65,
                )
            )
            ax.text(8.55, 0, "+", ha="center", va="center", fontsize=11)
            ax.annotate(
                "",
                xy=(8.32, 0),
                xytext=(7.53, 0),
                arrowprops={"arrowstyle": "->", "lw": 0.65},
            )
            ax.annotate(
                "",
                xy=(10, 0),
                xytext=(8.78, 0),
                arrowprops={"arrowstyle": "->", "lw": 0.65},
            )
            ax.text(0.1, -0.42, "Input", fontsize=8)
            ax.text(9.4, -0.42, "Output", fontsize=8)
            ax.plot([0.4, 0.4], [0, 0.9], color="black", linewidth=0.65)
            if index:
                ax.plot([0.4, 4.17], [0.9, 0.9], color="black", linewidth=0.65)
                for x, label in ((4.6, "LN"), (5.8, "Dense")):
                    ax.add_patch(
                        Rectangle(
                            (x - 0.43, 0.68),
                            0.86,
                            0.44,
                            facecolor="white",
                            edgecolor="black",
                            linewidth=0.65,
                        )
                    )
                    ax.text(x, 0.9, label, ha="center", va="center", fontsize=8)
                ax.annotate(
                    "",
                    xy=(5.37, 0.9),
                    xytext=(5.03, 0.9),
                    arrowprops={"arrowstyle": "->", "lw": 0.65},
                )
                ax.plot([6.23, 8.55], [0.9, 0.9], color="black", linewidth=0.65)
            else:
                ax.plot([0.4, 8.55], [0.9, 0.9], color="black", linewidth=0.65)
            ax.annotate(
                "",
                xy=(8.55, 0.23),
                xytext=(8.55, 0.9),
                arrowprops={"arrowstyle": "->", "lw": 0.65},
            )
        self.save("residual_block_single_column", fig, selected_dropout=0.0)

    def model_diagram(self):
        """Draw the selected model using widths read from its saved Keras graph."""
        path = self.run / "retrained_paper_model.keras"
        model_sha256 = digest(path)
        audit = json.loads((self.run / "architecture_audit.json").read_text())
        assert model_sha256 == audit["model_sha256"]
        with zipfile.ZipFile(path) as archive:
            graph = json.loads(archive.read("config.json"))["config"]["layers"]
        layers = {layer["config"]["name"]: layer for layer in graph}
        input_layer = next(
            layer for layer in graph if layer["class_name"] == "InputLayer"
        )
        input_width = input_layer["config"]["batch_shape"][-1]
        stem_width = layers["input_stem_dense"]["config"]["units"]
        shared_width = layers["trunk_final_ln"]["inbound_nodes"][0]["args"][0][
            "config"
        ]["shape"][-1]
        assert sum(layer["class_name"] == "Add" for layer in graph) == 1
        assert not any(layer["class_name"] == "Dropout" for layer in graph)
        heads = {}
        tasks = (
            ("teff_output", "Teff", r"$T_{\mathrm{eff}}$", 50),
            ("feh_output", "[Fe/H]", r"$[\mathrm{Fe/H}]$", 31),
            ("logg_output", "log g", r"$\log g$", 12),
        )
        for name, target, _, _ in tasks:
            hidden = [
                layer["config"]
                for layer in graph
                if layer["class_name"] == "Dense"
                and layer["config"]["name"].startswith(f"{name}_head_dense_")
            ]
            hidden.sort(key=lambda layer: int(layer["name"].rsplit("_", 1)[-1]))
            heads[target] = [layer["units"] for layer in hidden]
            assert all(layer["activation"] == "gelu" for layer in hidden)
            assert layers[name]["config"]["units"] == 1
            assert layers[name]["config"]["activation"] == "linear"
        assert heads == audit["actual_heads"]

        fig = plt.figure(figsize=(WIDTH, 68 / 25.4))
        ax = fig.add_axes([0.012, 0.035, 0.976, 0.93])
        ax.set_xlim(-1, 162)
        ax.set_ylim(-3, 70)
        ax.axis("off")

        def box(left, width, centre_y, label):
            ax.add_patch(
                Rectangle(
                    (left, centre_y - 7),
                    width,
                    14,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.65,
                )
            )
            ax.text(
                left + width / 2,
                centre_y,
                label,
                ha="center",
                va="center",
                fontsize=8,
                linespacing=1.25,
            )

        def arrow(start, end):
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "->",
                    "lw": 0.65,
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "mutation_scale": 7,
                },
            )

        ax.text(45, 65, "Shared backbone", ha="center", va="center", fontsize=9)
        ax.text(
            130.5,
            65,
            "Task-specific regression heads",
            ha="center",
            va="center",
            fontsize=9,
        )
        for left, right in ((0, 90), (100, 161)):
            ax.plot([left, right], [61, 61], color="black", linewidth=0.45)
        box(0, 17, 31, f"Spectrum\n{input_width} bins")
        box(21, 23, 31, f"Stem\nDense {stem_width}\nLN → GELU")
        box(
            48, 26, 31, f"Residual block\n{stem_width} → {shared_width}\nProjected skip"
        )
        box(78, 12, 31, f"LN\n{shared_width}")
        for left, right in ((17, 21), (44, 48), (74, 78)):
            arrow((left, 31), (right, 31))
        ax.plot([90, 95], [31, 31], color="black", linewidth=0.65)
        ax.plot([95, 95], [12, 50], color="black", linewidth=0.65)
        ax.plot(95, 31, "o", color="black", markersize=2)
        for _, target, output_label, centre_y in tasks:
            widths = " → ".join(str(width) for width in heads[target])
            activation = "GELU" if len(heads[target]) == 1 else "GELU after each layer"
            box(100, 37, centre_y, f"Dense {widths}\n{activation}")
            box(141, 20, centre_y, f"Linear 1\n{output_label}")
            arrow((95, centre_y), (100, centre_y))
            arrow((137, centre_y), (141, centre_y))
        ax.text(
            80.5,
            -1,
            f"{audit['trainable_parameters']:,} trainable parameters; dropout = 0",
            ha="center",
            va="center",
            fontsize=7.8,
        )
        self.save(
            "model_architecture_single_column",
            fig,
            model_sha256=model_sha256,
            input_features=input_width,
            stem_units=stem_width,
            residual_blocks=1,
            residual_units=shared_width,
            heads=heads,
            output_units_per_head=1,
            output_activation="linear",
            selected_dropout=0.0,
            trainable_parameters=audit["trainable_parameters"],
            source="Layer widths and activations read from the evaluated saved Keras graph",
        )

    def training(self):
        path = self.run / "history.csv"
        config_path = self.run / "run_config.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Training history for the evaluated checkpoint is missing: {path}"
            )
        frame = pd.read_csv(path)
        config = json.loads(config_path.read_text())
        epochs = frame.epoch.to_numpy() + 1
        selected = int(np.nanargmin(frame.val_normalized_mae_sum.to_numpy()))
        fig, axes = self.axes(rows=2, cols=2, height=96)
        plots = [
            ("loss", "Weighted Huber loss", "Total loss"),
            ("teff_output_mae", "Scaled MAE", r"$T_{\mathrm{eff}}$"),
            ("feh_output_mae", "Scaled MAE", r"$[\mathrm{Fe/H}]$"),
            ("logg_output_mae", "Scaled MAE", r"$\log g$"),
        ]
        for j, (ax, (key, ylabel, title)) in enumerate(zip(axes.ravel(), plots)):
            ax.plot(epochs, frame[key], color="black", label="Training")
            ax.plot(
                epochs,
                frame["val_" + key],
                color="black",
                linestyle="--",
                label="Validation",
            )
            ax.plot(
                epochs[selected],
                frame["val_" + key].iloc[selected],
                marker="*",
                color="black",
                markersize=5,
                label="Selected checkpoint",
            )
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True)
            ax.set_xlabel("Epoch" if j >= 2 else "")
            ax.set_xlim(1, epochs.max())
        axes[0, 0].legend(frameon=False, fontsize=7.3)
        self.save(
            "training_validation_single_column",
            fig,
            epochs=len(frame),
            selected_epoch=int(epochs[selected]),
            scope="Final training of the Bayesian-selected 530275-parameter architecture; same validation-selected checkpoint as all reported test metrics",
            retraining_test_mae=config.get("test_mae"),
        )
        self.manifest["notes"] = [
            note
            for note in self.manifest["notes"]
            if not note.startswith("Training figure pending")
        ]

    def finish(self):
        (self.out / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )
        images = sorted(self.out.glob("*_single_column.png"))
        columns = 3
        cell = (600, 430)
        sheet = Image.new(
            "RGB",
            (columns * cell[0], ((len(images) + columns - 1) // columns) * cell[1]),
            "#eeeeee",
        )
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(images):
            image = Image.open(path).convert("RGB")
            image = ImageOps.contain(image, (580, 380))
            x = (i % columns) * cell[0]
            y = (i // columns) * cell[1]
            draw.text(
                (x + 8, y + 8), path.stem.replace("_single_column", ""), fill="black"
            )
            sheet.paste(
                image,
                (x + (cell[0] - image.width) // 2, y + 35 + (380 - image.height) // 2),
            )
        sheet.save(self.out / "preview.png")
        if self.paper:
            target = self.paper / "figs/single_column"
            target.mkdir(parents=True, exist_ok=True)
            for path in self.out.glob("*_single_column.*"):
                if path.suffix in (".pdf", ".png"):
                    shutil.copy2(path, target / path.name)
            repro = self.paper / "reproducibility"
            repro.mkdir(exist_ok=True)
            for name in (
                "manifest.json",
                "snr_error_statistics.csv",
                "optimization_scores.csv",
            ):
                if (self.out / name).exists():
                    shutil.copy2(self.out / name, repro / name)
            for name in ("run_config.json", "history.csv", "history.json"):
                source = self.run / name
                if source.exists():
                    shutil.copy2(source, repro / f"retraining_{name}")
            for name in (
                "bootstrap_mae.json",
                "paired_bootstrap.json",
                "complexity.csv",
                "architecture_audit.json",
                "parameter_range_errors.csv",
                "parameter_range_errors.json",
                "benchmark_configuration.json",
                "software_versions.json",
                "research_audit.json",
                "research_audit.md",
            ):
                source = self.run / name
                if source.exists():
                    shutil.copy2(source, repro / name)
            source = self.run / "ablations/ablation_summary.csv"
            if source.exists():
                shutil.copy2(source, repro / "ablation_summary.csv")
            source = self.run / "ablations/ablation_paired_bootstrap.json"
            if source.exists():
                shutil.copy2(source, repro / "ablation_paired_bootstrap.json")
            for source in (self.run / "ablations").glob("*/metrics.json"):
                shutil.copy2(source, repro / f"ablation_{source.parent.name}.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--paper-dir", type=Path)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="One experiment supplying weights, predictions and complete training history.",
    )
    parser.add_argument("--only-training", action="store_true")
    args = parser.parse_args()
    out = args.out_dir or args.project_root / "outputs/paper_figures/single_column"
    figures = Figures(args.project_root, out, args.paper_dir, args.run_dir)
    if args.only_training:
        previous = out / "manifest.json"
        if previous.exists():
            figures.manifest = json.loads(previous.read_text(encoding="utf-8"))
        if figures.manifest.get("evaluated_model", {}).get("sha256") != digest(
            figures.run / "retrained_paper_model.keras"
        ):
            raise ValueError(
                "The evaluated checkpoint changed; regenerate every figure without --only-training."
            )
        figures.training()
    else:
        figures.load_metadata()
        figures.sky()
        figures.observations()
        figures.distributions()
        figures.raw_diagnostics()
        figures.optimization()
        figures.weights()
        figures.predictions()
        figures.residual_diagram()
        figures.model_diagram()
        figures.training()
        from analyze_parameter_ranges import analyze

        analyze(figures)
    figures.finish()


if __name__ == "__main__":
    main()
