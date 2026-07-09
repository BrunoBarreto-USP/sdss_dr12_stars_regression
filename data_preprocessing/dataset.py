"""Shared preprocessing loaders for the HF Parquet SDSS benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data_acquisition.hf_data import (
    CANONICAL_SPLITS,
    RAW_ARRAY_COLUMNS,
    TARGET_COLUMNS,
    load_split_dataframe,
    split_to_arrays,
    stack_array_column,
    target_array,
)


def normalize_flux_per_spectrum(flux: np.ndarray) -> np.ndarray:
    """Standardize each spectrum independently."""
    flux = np.asarray(flux, dtype=np.float32)
    mean = flux.mean(axis=1, keepdims=True)
    std = flux.std(axis=1, keepdims=True)
    std = np.where(std > 0.0, std, 1.0)
    return ((flux - mean) / std).astype(np.float32)


def load_processed_split(split: str, *, row_limit: int | None = None, data_dir: str | Path | None = None):
    kwargs = {"row_limit": row_limit}
    if data_dir is not None:
        kwargs["data_dir"] = data_dir
    return split_to_arrays(split, use_processed_flux=True, include_raw=False, **kwargs)


def load_raw_split(split: str, *, row_limit: int | None = None, data_dir: str | Path | None = None):
    kwargs = {"row_limit": row_limit}
    if data_dir is not None:
        kwargs["data_dir"] = data_dir
    return split_to_arrays(split, use_processed_flux=False, include_raw=True, **kwargs)
