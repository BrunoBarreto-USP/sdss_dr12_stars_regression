"""Verify that a compact training cache still matches its local Parquet splits."""

from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TARGET_COLUMNS = ("catalog_teff", "catalog_feh", "catalog_logg")


def validate_compact_dataset(
    data, data_dir: Path, *, row_limit: int | None = None
) -> dict[str, int]:
    if row_limit is not None and row_limit < 1:
        raise ValueError("row_limit must be positive.")
    center = np.asarray(data["label_robust_center"], dtype=np.float64)
    scale = np.asarray(data["label_robust_scale"], dtype=np.float64)
    if (
        center.shape != (3,)
        or scale.shape != (3,)
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("Invalid target scaler dimensions or values.")
    identifiers = []
    counts = {}
    for split, short in (("train", "train"), ("validation", "val"), ("test", "test")):
        key = f"X_{short}_features"
        if key not in data:
            raise ValueError(f"Missing {split} feature array.")
        features = np.asarray(data[key])
        if features.ndim != 2 or features.shape[1] != 4000 or features.dtype != np.float32 or not np.isfinite(features).all():
            raise ValueError(f"Invalid {split} features: require finite float32 (N, 4000).")
        frame = pd.read_parquet(
            Path(data_dir) / f"{split}.parquet", columns=["spec_id", *TARGET_COLUMNS]
        )
        if row_limit is not None:
            frame = frame.head(row_limit)
        ids = np.asarray(data[f"{short}_ids"])
        if ids.ndim != 1 or ids.dtype != np.int64:
            raise ValueError(f"Invalid {split} identifier type; require int64 ordered IDs.")
        if len(features) != len(ids):
            raise ValueError(f"Feature/identifier counts differ for {split}.")
        np.testing.assert_array_equal(
            ids, frame.spec_id.to_numpy(), err_msg=f"Stale {split} compact cache IDs."
        )
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"Repeated spectrum IDs in {split}.")
        labels = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32)
        # Compare every processed input, not only labels and identifiers.
        if "processed_flux" in pq.ParquetFile(Path(data_dir) / f"{split}.parquet").schema_arrow.names:
            expected = hashlib.sha256()
            remaining = len(features)
            normalize = bool(data["flux_normalized_per_spectrum"]) if "flux_normalized_per_spectrum" in data else False
            for batch in pq.ParquetFile(Path(data_dir) / f"{split}.parquet").iter_batches(batch_size=256, columns=["processed_flux"]):
                column = batch.column(0)
                lengths = column.value_lengths().to_numpy(zero_copy_only=False)
                if not np.all(lengths == 4000):
                    raise ValueError(f"Invalid {split} processed feature lengths.")
                chunk = column.flatten().to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(batch.num_rows, 4000)
                chunk = chunk[:remaining]
                if normalize:
                    from data_preprocessing.dataset import normalize_flux_per_spectrum
                    chunk = normalize_flux_per_spectrum(chunk)
                expected.update(memoryview(np.ascontiguousarray(chunk)).cast("B"))
                remaining -= len(chunk)
                if remaining == 0:
                    break
            actual_hash = hashlib.sha256(memoryview(np.ascontiguousarray(features)).cast("B")).hexdigest()
            if remaining or expected.hexdigest() != actual_hash:
                raise ValueError(f"Stale {split} feature values or wavelength ordering.")
        actual = np.asarray(data[f"y_{short}_targets"])
        if actual.dtype != np.float32 or actual.shape != labels.shape or not np.isfinite(actual).all():
            raise ValueError(f"Invalid scaled target matrix for {split}.")
        np.testing.assert_allclose(
            actual,
            (labels.astype(np.float64) - center) / scale,
            atol=2e-6,
            rtol=1e-5,
            err_msg=f"Stale {split} target values.",
        )
        if split == "train":
            q = np.percentile(labels, [25, 50, 75], axis=0)
            expected_scale = q[2] - q[0]
            expected_scale[expected_scale == 0] = 1
            np.testing.assert_allclose(
                center,
                q[1],
                atol=1e-6,
                rtol=1e-7,
                err_msg="Scaler median must come from training labels.",
            )
            np.testing.assert_allclose(
                scale,
                expected_scale,
                atol=1e-6,
                rtol=1e-7,
                err_msg="Scaler IQR must come from training labels.",
            )
        identifiers.append(set(ids.tolist()))
        counts[split] = len(ids)
    if any(identifiers[i] & identifiers[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Spectrum IDs overlap across experimental splits.")
    return counts
