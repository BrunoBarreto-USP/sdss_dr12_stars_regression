"""Plot one processed spectrum and label distributions from the HF benchmark."""

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

from data_exploration.common import FIGS_DIR, load_columns, processed_flux


TARGET_COLUMNS = ("catalog_teff", "catalog_feh", "catalog_logg")
WAVELENGTH_START = 3762.0
WAVELENGTH_END = 9268.0


def main(split: str = "train", spectrum_index: int = 123, row_limit: int | None = None) -> None:
    label_limit = row_limit
    spectrum_limit = max(spectrum_index + 1, row_limit or 0) if row_limit is not None else spectrum_index + 1
    labels_frame = load_columns(["spec_id", *TARGET_COLUMNS], split=split, row_limit=label_limit)
    spectrum_frame = load_columns(["processed_flux", "spec_id", *TARGET_COLUMNS], split=split, row_limit=spectrum_limit)
    flux = processed_flux(spectrum_frame)
    if not (0 <= spectrum_index < len(flux)):
        raise IndexError(f"spectrum_index must satisfy 0 <= index < {len(flux)}")

    wavelength = np.linspace(WAVELENGTH_START, WAVELENGTH_END, flux.shape[1])
    teff = labels_frame["catalog_teff"].to_numpy(dtype=np.float32)
    feh = labels_frame["catalog_feh"].to_numpy(dtype=np.float32)
    logg = labels_frame["catalog_logg"].to_numpy(dtype=np.float32)
    selected = spectrum_frame.iloc[spectrum_index]
    selected_teff = float(selected["catalog_teff"])
    selected_feh = float(selected["catalog_feh"])
    selected_logg = float(selected["catalog_logg"])
    spec_id = int(selected["spec_id"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=160)
    axes[0, 0].plot(wavelength, flux[spectrum_index], linewidth=1.0)
    axes[0, 0].set_title("Selected processed spectrum")
    axes[0, 0].set_xlabel("Wavelength [Angstrom]")
    axes[0, 0].set_ylabel("Processed flux")
    axes[0, 0].grid(True, alpha=0.3)

    for ax, values, selected, title, xlabel, color in [
        (axes[0, 1], teff, selected_teff, "Teff distribution", "Teff", "tab:blue"),
        (axes[1, 0], feh, selected_feh, "[Fe/H] distribution", "[Fe/H]", "tab:orange"),
        (axes[1, 1], logg, selected_logg, "logg distribution", "logg", "tab:green"),
    ]:
        ax.hist(values, bins=50, color=color, alpha=0.8)
        ax.axvline(selected, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")

    fig.suptitle(
        f"split={split} | row={spectrum_index} | spec_id={spec_id} | "
        f"teff={selected_teff:.2f} | feh={selected_feh:.3f} | logg={selected_logg:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGS_DIR / "preprocessed_spectrum_example.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot one processed spectrum from the HF benchmark.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--spectrum-index", type=int, default=123)
    parser.add_argument("--row-limit", type=int, default=None)
    args = parser.parse_args()
    main(split=args.split, spectrum_index=args.spectrum_index, row_limit=args.row_limit)
