"""Plot stellar-label coverage for the HF Parquet benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_exploration.common import FIGS_DIR, load_all_split_columns


LABEL_COLUMNS = ("catalog_teff", "catalog_feh", "catalog_logg")
LABELS = ("Teff", "[Fe/H]", "logg")


def main(row_limit_per_split: int | None = None) -> None:
    frame = load_all_split_columns(list(LABEL_COLUMNS), row_limit_per_split=row_limit_per_split)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=160)
    for ax, column, label in zip(axes, LABEL_COLUMNS, LABELS):
        ax.hist(frame[column], bins=60, color="#5d7c3b", alpha=0.85)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    out = FIGS_DIR / "label_coverage.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot label coverage for the HF benchmark.")
    parser.add_argument("--row-limit-per-split", type=int, default=None)
    args = parser.parse_args()
    main(row_limit_per_split=args.row_limit_per_split)
