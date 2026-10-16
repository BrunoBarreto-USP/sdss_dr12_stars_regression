"""Verify that a compact training cache still matches its local Parquet splits."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

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
        frame = pd.read_parquet(
            Path(data_dir) / f"{split}.parquet", columns=["spec_id", *TARGET_COLUMNS]
        )
        if row_limit is not None:
            frame = frame.head(row_limit)
        ids = np.asarray(data[f"{short}_ids"])
        np.testing.assert_array_equal(
            ids, frame.spec_id.to_numpy(), err_msg=f"Stale {split} compact cache IDs."
        )
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"Repeated spectrum IDs in {split}.")
        labels = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32)
        actual = np.asarray(data[f"y_{short}_targets"])
        if actual.shape != labels.shape or not np.isfinite(actual).all():
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
