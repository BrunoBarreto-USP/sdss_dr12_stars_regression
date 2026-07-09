"""Build a compact NPZ benchmark from the canonical HF Parquet dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_acquisition.hf_data import CANONICAL_SPLITS, TARGET_COLUMNS, split_to_arrays
from data_preprocessing.dataset import normalize_flux_per_spectrum


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "sdss_dr12_processed_flux_benchmark.npz"


def build_final_dataset(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    row_limit: int | None = None,
    normalize_flux: bool = False,
    overwrite: bool = False,
) -> Path:
    """Create a processed-flux NPZ with scaled targets for model training."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(f"Compact NPZ already exists, skipping: {output_path.resolve()}")
        return output_path

    loaded = {split: split_to_arrays(split, row_limit=row_limit) for split in CANONICAL_SPLITS}

    X_train = np.asarray(loaded["train"]["processed_flux"], dtype=np.float32)
    X_val = np.asarray(loaded["validation"]["processed_flux"], dtype=np.float32)
    X_test = np.asarray(loaded["test"]["processed_flux"], dtype=np.float32)
    if normalize_flux:
        X_train = normalize_flux_per_spectrum(X_train)
        X_val = normalize_flux_per_spectrum(X_val)
        X_test = normalize_flux_per_spectrum(X_test)

    y_train_original = np.asarray(loaded["train"]["targets"], dtype=np.float32)
    y_val_original = np.asarray(loaded["validation"]["targets"], dtype=np.float32)
    y_test_original = np.asarray(loaded["test"]["targets"], dtype=np.float32)

    scaler = RobustScaler()
    y_train = scaler.fit_transform(y_train_original).astype(np.float32)
    y_val = scaler.transform(y_val_original).astype(np.float32)
    y_test = scaler.transform(y_test_original).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_train_features=X_train,
        X_val_features=X_val,
        X_test_features=X_test,
        y_train_targets=y_train,
        y_val_targets=y_val,
        y_test_targets=y_test,
        y_train_targets_original=y_train_original,
        y_val_targets_original=y_val_original,
        y_test_targets_original=y_test_original,
        train_ids=loaded["train"]["metadata"]["spec_id"].to_numpy(dtype=np.int64),
        val_ids=loaded["validation"]["metadata"]["spec_id"].to_numpy(dtype=np.int64),
        test_ids=loaded["test"]["metadata"]["spec_id"].to_numpy(dtype=np.int64),
        target_names=np.asarray(["Teff", "FeH", "logg"]),
        target_columns=np.asarray(TARGET_COLUMNS),
        label_robust_center=scaler.center_.astype(np.float32),
        label_robust_scale=scaler.scale_.astype(np.float32),
        flux_normalized_per_spectrum=np.asarray(bool(normalize_flux)),
        source_dataset=np.asarray("BrunoBarreto/sdss_dr12_stars_regression"),
    )
    print(f"Saved {output_path.resolve()}")
    print(f"train={X_train.shape}, validation={X_val.shape}, test={X_test.shape}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact NPZ from HF Parquet splits.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--normalize-flux", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild even if the output NPZ already exists.")
    args = parser.parse_args()
    build_final_dataset(
        output_path=args.out,
        row_limit=args.row_limit,
        normalize_flux=args.normalize_flux,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
