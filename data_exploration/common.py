"""Shared data loading for exploration scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_acquisition.hf_data import CANONICAL_SPLITS, load_split_dataframe, stack_array_column


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGS_DIR = PROJECT_ROOT / "figs"


def load_columns(columns: list[str], *, split: str = "train", row_limit: int | None = None) -> pd.DataFrame:
    return load_split_dataframe(split, columns=columns, row_limit=row_limit)


def load_all_split_columns(columns: list[str], *, row_limit_per_split: int | None = None) -> pd.DataFrame:
    frames = []
    for split in CANONICAL_SPLITS:
        frame = load_split_dataframe(split, columns=columns, row_limit=row_limit_per_split)
        frame.insert(0, "split", split)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def processed_flux(frame: pd.DataFrame) -> np.ndarray:
    return stack_array_column(frame, "processed_flux", dtype=np.float32)
