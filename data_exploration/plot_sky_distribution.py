"""Plot sky coverage from the HF benchmark metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_exploration.common import FIGS_DIR, load_all_split_columns


def main(row_limit_per_split: int | None = None) -> None:
    frame = load_all_split_columns(["ra", "dec"], row_limit_per_split=row_limit_per_split)
    ra_deg = frame["ra"].to_numpy(dtype=np.float64)
    dec_deg = frame["dec"].to_numpy(dtype=np.float64)
    valid = np.isfinite(ra_deg) & np.isfinite(dec_deg)
    ra_deg = ra_deg[valid]
    dec_deg = dec_deg[valid]
    if len(ra_deg) < 2:
        raise RuntimeError("Need at least two valid sky positions to plot.")

    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    ra_rad = np.remainder(ra_rad + 2.0 * np.pi, 2.0 * np.pi)
    ra_rad[ra_rad > np.pi] -= 2.0 * np.pi
    ra_rad = -ra_rad

    xy = np.vstack([ra_rad, dec_rad])
    density = gaussian_kde(xy)(xy)
    order = density.argsort()
    ra_rad, dec_rad, density = ra_rad[order], dec_rad[order], density[order]

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="aitoff")
    sc = ax.scatter(ra_rad, dec_rad, c=density, s=3, cmap="viridis", alpha=0.9, linewidths=0, rasterized=True)
    ax.grid(True, alpha=0.35)
    cbar = plt.colorbar(sc, ax=ax, pad=0.06, shrink=0.72)
    cbar.set_label("Relative local density")
    out = FIGS_DIR / "sky_distribution_density_aitoff.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sky distribution for the HF benchmark.")
    parser.add_argument("--row-limit-per-split", type=int, default=None)
    args = parser.parse_args()
    main(row_limit_per_split=args.row_limit_per_split)
