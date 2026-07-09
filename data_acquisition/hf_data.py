"""Hugging Face dataset helpers for the SDSS DR12 regression benchmark.

The canonical public dataset is Parquet-based and contains one file per split:
``train.parquet``, ``validation.parquet``, and ``test.parquet``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

HF_DATASET_REPO = "BrunoBarreto/sdss_dr12_stars_regression"
SPLIT_FILES = {
    "train": "train.parquet",
    "validation": "validation.parquet",
    "val": "validation.parquet",
    "test": "test.parquet",
}
CANONICAL_SPLITS = ("train", "validation", "test")

PROCESSED_FLUX_COLUMN = "processed_flux"
RAW_ARRAY_COLUMNS = ("flux", "loglam", "ivar", "mask")
TARGET_COLUMNS = ("catalog_teff", "catalog_feh", "catalog_logg")
UNCERTAINTY_COLUMNS = ("catalog_teff_unc", "catalog_feh_unc", "catalog_logg_unc")
METADATA_COLUMNS = ("spec_id", "plate", "mjd", "fiber", "ra", "dec", "snr", "rv_adop")
REQUIRED_COLUMNS = METADATA_COLUMNS + RAW_ARRAY_COLUMNS + (PROCESSED_FLUX_COLUMN,) + TARGET_COLUMNS

LEGACY_SPLIT_FILENAME = "split_from_large_test_30k_5k_15k.npz"
LEGACY_SCALERS_FILENAME = "target_scalers_split_from_large_test_30k_5k_15k.npz"


def normalize_split_name(split: str) -> str:
    """Return the canonical split name used by the dataset."""
    split_name = str(split).lower()
    if split_name == "val":
        return "validation"
    if split_name not in CANONICAL_SPLITS:
        raise ValueError(f"Unknown split {split!r}. Expected one of {CANONICAL_SPLITS}.")
    return split_name


def split_filename(split: str) -> str:
    """Return the Parquet filename for *split*."""
    return SPLIT_FILES[str(split).lower()]


def ensure_parquet_split(split: str, data_dir: str | Path = DEFAULT_DATA_DIR) -> Path:
    """Download/cache a split Parquet file and return its local path."""
    split_name = normalize_split_name(split)
    local_path = Path(data_dir) / split_filename(split_name)
    if local_path.exists():
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        filename=split_filename(split_name),
        local_dir=local_path.parent,
        local_dir_use_symlinks=False,
    )
    return Path(cached)


def ensure_all_parquet_splits(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, Path]:
    """Download/cache all canonical Parquet splits."""
    return {split: ensure_parquet_split(split, data_dir=data_dir) for split in CANONICAL_SPLITS}


def _read_parquet(path: Path, columns: Iterable[str] | None, row_limit: int | None) -> pd.DataFrame:
    if row_limit is None:
        return pd.read_parquet(path, columns=list(columns) if columns is not None else None)

    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=list(columns) if columns is not None else None)
        return table.slice(0, int(row_limit)).to_pandas()
    except Exception:
        frame = pd.read_parquet(path, columns=list(columns) if columns is not None else None)
        return frame.head(int(row_limit)).copy()


def load_split_dataframe(
    split: str,
    *,
    columns: Iterable[str] | None = None,
    row_limit: int | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Load a dataset split as a pandas DataFrame."""
    split_name = normalize_split_name(split)
    local_path = Path(data_dir) / split_filename(split_name)
    if local_path.exists():
        frame = _read_parquet(local_path, columns=columns, row_limit=row_limit)
    elif row_limit is not None:
        try:
            from datasets import load_dataset

            stream = load_dataset(HF_DATASET_REPO, split=split_name, streaming=True)
            rows = []
            selected_columns = list(columns) if columns is not None else None
            for row_idx, row in enumerate(stream):
                if row_idx >= int(row_limit):
                    break
                if selected_columns is not None:
                    row = {column: row[column] for column in selected_columns}
                rows.append(row)
            frame = pd.DataFrame(rows)
        except Exception:
            path = ensure_parquet_split(split_name, data_dir=data_dir)
            frame = _read_parquet(path, columns=columns, row_limit=row_limit)
    else:
        path = ensure_parquet_split(split_name, data_dir=data_dir)
        frame = _read_parquet(path, columns=columns, row_limit=row_limit)
    missing = [column for column in (columns or []) if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns in {path.name}: {missing}")
    return frame


def require_columns(frame: pd.DataFrame, columns: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Validate that *frame* contains all required benchmark columns."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required dataset columns: {missing}")


def stack_array_column(frame: pd.DataFrame, column: str, *, dtype=np.float32) -> np.ndarray:
    """Stack an array-valued column, preserving ragged arrays when needed."""
    values = [np.asarray(value, dtype=dtype) for value in frame[column]]
    if not values:
        return np.empty((0, 0), dtype=dtype)
    try:
        return np.stack(values).astype(dtype, copy=False)
    except ValueError:
        return np.asarray(values, dtype=object)


def target_array(frame: pd.DataFrame, *, dtype=np.float32) -> np.ndarray:
    """Return targets in ``catalog_teff, catalog_feh, catalog_logg`` order."""
    return frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=dtype)


def metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return available metadata columns from a split DataFrame."""
    columns = [column for column in METADATA_COLUMNS if column in frame.columns]
    return frame.loc[:, columns].copy()


def split_to_arrays(
    split: str,
    *,
    use_processed_flux: bool = True,
    include_raw: bool = False,
    row_limit: int | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> dict[str, np.ndarray | pd.DataFrame]:
    """Load one split as arrays used by training and preprocessing scripts."""
    columns = list(METADATA_COLUMNS) + list(TARGET_COLUMNS)
    if use_processed_flux:
        columns.append(PROCESSED_FLUX_COLUMN)
    if include_raw:
        columns.extend(RAW_ARRAY_COLUMNS)

    frame = load_split_dataframe(split, columns=columns, row_limit=row_limit, data_dir=data_dir)
    arrays: dict[str, np.ndarray | pd.DataFrame] = {
        "targets": target_array(frame),
        "metadata": metadata_frame(frame),
    }
    if use_processed_flux:
        arrays["processed_flux"] = stack_array_column(frame, PROCESSED_FLUX_COLUMN)
    if include_raw:
        for column in RAW_ARRAY_COLUMNS:
            dtype = bool if column == "mask" else np.float32
            arrays[column] = stack_array_column(frame, column, dtype=dtype)
    return arrays


def load_benchmark_arrays(
    *,
    row_limit: int | None = None,
    scale_targets: bool = True,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> dict[str, np.ndarray]:
    """Return old-style benchmark arrays backed by the Parquet dataset."""
    from sklearn.preprocessing import RobustScaler

    loaded = {
        split: split_to_arrays(split, row_limit=row_limit, data_dir=data_dir)
        for split in CANONICAL_SPLITS
    }
    y_train = np.asarray(loaded["train"]["targets"], dtype=np.float32)
    y_val = np.asarray(loaded["validation"]["targets"], dtype=np.float32)
    y_test = np.asarray(loaded["test"]["targets"], dtype=np.float32)

    if scale_targets:
        scaler = RobustScaler()
        y_train_out = scaler.fit_transform(y_train).astype(np.float32)
        y_val_out = scaler.transform(y_val).astype(np.float32)
        y_test_out = scaler.transform(y_test).astype(np.float32)
        centers = scaler.center_.astype(np.float32)
        scales = scaler.scale_.astype(np.float32)
    else:
        y_train_out, y_val_out, y_test_out = y_train, y_val, y_test
        centers = np.zeros(len(TARGET_COLUMNS), dtype=np.float32)
        scales = np.ones(len(TARGET_COLUMNS), dtype=np.float32)

    return {
        "X_train_features": np.asarray(loaded["train"]["processed_flux"], dtype=np.float32),
        "X_val_features": np.asarray(loaded["validation"]["processed_flux"], dtype=np.float32),
        "X_test_features": np.asarray(loaded["test"]["processed_flux"], dtype=np.float32),
        "y_train_targets": y_train_out,
        "y_val_targets": y_val_out,
        "y_test_targets": y_test_out,
        "train_ids": loaded["train"]["metadata"]["spec_id"].to_numpy(dtype=np.int64),
        "val_ids": loaded["validation"]["metadata"]["spec_id"].to_numpy(dtype=np.int64),
        "test_ids": loaded["test"]["metadata"]["spec_id"].to_numpy(dtype=np.int64),
        "target_names": np.asarray(["Teff", "FeH", "logg"]),
        "centers": centers,
        "scales": scales,
    }


def save_legacy_npz_files(data_dir: str | Path = DEFAULT_DATA_DIR) -> tuple[Path, Path]:
    """Materialize compatibility NPZ files from the canonical Parquet dataset."""
    data_path = Path(data_dir) / LEGACY_SPLIT_FILENAME
    scalers_path = Path(data_dir) / LEGACY_SCALERS_FILENAME
    if data_path.exists() and scalers_path.exists():
        return data_path, scalers_path

    arrays = load_benchmark_arrays(data_dir=data_dir)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_path,
        X_train_features=arrays["X_train_features"],
        X_val_features=arrays["X_val_features"],
        X_test_features=arrays["X_test_features"],
        y_train_targets=arrays["y_train_targets"],
        y_val_targets=arrays["y_val_targets"],
        y_test_targets=arrays["y_test_targets"],
        train_ids=arrays["train_ids"],
        val_ids=arrays["val_ids"],
        test_ids=arrays["test_ids"],
        target_names=arrays["target_names"],
    )
    np.savez_compressed(
        scalers_path,
        centers=arrays["centers"],
        scales=arrays["scales"],
        target_names=arrays["target_names"],
    )
    return data_path, scalers_path


def ensure_hf_asset(path: str | Path, *, required: bool = False) -> Path:
    """Compatibility wrapper for older scripts.

    New code should call ``ensure_parquet_split`` or ``load_split_dataframe``.
    When older scripts request the historical NPZ files, they are generated from
    the Parquet benchmark.
    """
    resolved = Path(path)
    if resolved.exists():
        return resolved

    if resolved.name in {LEGACY_SPLIT_FILENAME, LEGACY_SCALERS_FILENAME}:
        data_path, scalers_path = save_legacy_npz_files(resolved.parent)
        return data_path if resolved.name == LEGACY_SPLIT_FILENAME else scalers_path

    if resolved.name in set(SPLIT_FILES.values()):
        split = next(name for name, filename in SPLIT_FILES.items() if filename == resolved.name)
        return ensure_parquet_split(split, data_dir=resolved.parent)

    if required:
        raise FileNotFoundError(f"No Hugging Face dataset asset configured for {resolved.name}")
    return resolved
