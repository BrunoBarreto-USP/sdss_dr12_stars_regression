"""Plot the catalog SNR distribution from the HF benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_exploration.common import FIGS_DIR, load_all_split_columns


def main(row_limit_per_split: int | None = None) -> None:
    frame = load_all_split_columns(["snr"], row_limit_per_split=row_limit_per_split)
    snr = frame["snr"].to_numpy(dtype=np.float64)
    snr = snr[np.isfinite(snr)]
    if snr.size == 0:
        raise RuntimeError("No finite SNR values found in the HF dataset.")

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.hist(snr, bins=60, color="#2f6f9f", alpha=0.85)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Number of spectra")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    out = FIGS_DIR / "snr_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot SNR distribution for the HF benchmark.")
    parser.add_argument("--row-limit-per-split", type=int, default=None)
    args = parser.parse_args()
    main(row_limit_per_split=args.row_limit_per_split)
