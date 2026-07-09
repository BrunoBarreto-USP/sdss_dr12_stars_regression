"""Summarize catalog label uncertainties from the HF benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_exploration.common import load_all_split_columns


COLUMNS = (
    "catalog_teff",
    "catalog_teff_unc",
    "catalog_feh",
    "catalog_feh_unc",
    "catalog_logg",
    "catalog_logg_unc",
)


def _relative(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        np.abs(denominator),
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=np.abs(denominator) > 0,
    )


def main(row_limit_per_split: int | None = None) -> None:
    frame = load_all_split_columns(list(COLUMNS), row_limit_per_split=row_limit_per_split)
    teff = frame["catalog_teff"].to_numpy(dtype=np.float64)
    teff_unc = frame["catalog_teff_unc"].to_numpy(dtype=np.float64)
    feh = frame["catalog_feh"].to_numpy(dtype=np.float64)
    feh_unc = frame["catalog_feh_unc"].to_numpy(dtype=np.float64)
    logg = frame["catalog_logg"].to_numpy(dtype=np.float64)
    logg_unc = frame["catalog_logg_unc"].to_numpy(dtype=np.float64)

    print(f"rows analyzed: {len(frame)}")
    print(f"catalog_teff_unc mean: {np.nanmean(teff_unc):.6f} ({100.0 * np.nanmean(_relative(teff_unc, teff)):.4f}% relative)")
    print(f"catalog_feh_unc mean: {np.nanmean(feh_unc):.6f} ({100.0 * np.nanmean(_relative(feh_unc, feh)):.4f}% relative)")
    print(f"catalog_logg_unc mean: {np.nanmean(logg_unc):.6f} ({100.0 * np.nanmean(_relative(logg_unc, logg)):.4f}% relative)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize catalog label uncertainties without filtering rows.")
    parser.add_argument("--row-limit-per-split", type=int, default=None)
    args = parser.parse_args()
    main(row_limit_per_split=args.row_limit_per_split)
