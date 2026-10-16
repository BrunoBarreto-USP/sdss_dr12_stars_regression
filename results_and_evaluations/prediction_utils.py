"""Canonical target order for NumPy, list and named Keras outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np

OUTPUT_NAMES = ("teff_output", "feh_output", "logg_output")


def prediction_matrix(raw, output_names: Sequence[str] = OUTPUT_NAMES) -> np.ndarray:
    names = tuple(output_names)
    if isinstance(raw, Mapping):
        if set(raw) != set(names):
            raise ValueError(f"Prediction names {tuple(raw)} do not match {names}.")
        matrix = np.column_stack([np.asarray(raw[name]).reshape(-1) for name in names])
    elif isinstance(raw, (list, tuple)):
        if len(raw) != len(names):
            raise ValueError("Prediction count does not match target count.")
        matrix = np.column_stack([np.asarray(value).reshape(-1) for value in raw])
    else:
        matrix = np.asarray(raw)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(names)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Expected a finite N-by-targets prediction matrix.")
    return matrix
