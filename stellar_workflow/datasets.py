"""Pinned public data, separate sample caches and fingerprinted compact inputs."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import RobustScaler

from .config import ROOT, SPLITS, TARGET_NAMES
from .runs import sha256, write_json

REFERENCE = json.loads((ROOT / "reference/dataset.json").read_text(encoding="utf-8"))
GRID = REFERENCE["grid"]
COMPACT_NAME = "sdss_dr12_processed_flux_benchmark.npz"
TARGET_COLUMNS = ("catalog_teff", "catalog_feh", "catalog_logg")


def wavelength_grid():
    return 10 ** np.linspace(GRID["log10_start"], GRID["log10_end"], GRID["points"])


def fingerprint(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def download_verified(url, destination, expected_sha256, *, offline=False, attempts=3):
    """Ordinary HTTPS avoids Xet-specific failures; only verified bytes become a cache."""
    destination = Path(destination)
    if destination.exists():
        if sha256(destination) != expected_sha256:
            raise ValueError(f"Checksum mismatch in existing cache: {destination}; preserve it and choose a fresh cache directory.")
        return destination
    if offline:
        raise FileNotFoundError(f"Offline cache is missing: {destination}")
    import requests
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    error = None
    for attempt in range(attempts):
        try:
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                digest = hashlib.sha256()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                            digest.update(chunk)
                if digest.hexdigest() != expected_sha256:
                    raise ValueError(f"Downloaded checksum does not match the pinned file: {destination.name}")
            temporary.replace(destination)
            return destination
        except (requests.RequestException, OSError, ValueError) as exc:
            error = exc
            temporary.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 4))
    raise RuntimeError(f"Verified download failed after {attempts} attempts: {destination.name}: {error}") from error


def _manifest_path(path):
    return path.parent / ("compact_manifest.json" if path.name == COMPACT_NAME else path.stem + ".manifest.json")


def source_files(data_dir):
    root = Path(data_dir)
    return {
        split: {"sha256": sha256(root / f"{split}.parquet"),
                "rows": pq.ParquetFile(root / f"{split}.parquet").metadata.num_rows}
        for split, _ in SPLITS
    }


def build_compact(data_dir, *, source_kind="public", row_limit=None, normalize_flux=False, output_path=None, overwrite=False):
    from data_acquisition.hf_data import split_to_arrays
    from data_preprocessing.dataset import normalize_flux_per_spectrum
    from data_preprocessing.validation import validate_compact_dataset
    root = Path(data_dir).resolve()
    path = Path(output_path or root / COMPACT_NAME).resolve()
    if row_limit is not None and row_limit < 1:
        raise ValueError("row_limit must be positive.")
    if path.exists() and not overwrite:
        previous = json.loads(_manifest_path(path).read_text(encoding="utf-8")) if _manifest_path(path).exists() else {}
        if previous.get("row_limit") != row_limit or previous.get("source_kind") != source_kind:
            raise ValueError("Cached source kind or row limit differs; use a fresh cache directory.")
        with load_compact(path, data_dir=root) as existing:
            if bool(existing["flux_normalized_per_spectrum"]) != normalize_flux:
                raise ValueError("Normalization differs from the cached experiment; use a fresh cache.")
        return path
    files = source_files(root)
    if source_kind == "public" and any(files[s] != REFERENCE["splits"][s] for s, _ in SPLITS):
        raise ValueError("Public source files do not match the pinned dataset checksums/counts.")
    source = {"schema_version": 1, "source_kind": source_kind,
              "repo_id": REFERENCE["repo_id"] if source_kind.startswith("public") else None,
              "revision": REFERENCE["revision"] if source_kind.startswith("public") else None,
              "split_files": files, "row_limit": row_limit, "grid": GRID,
              "normalize_flux": normalize_flux, "reference_labels": "inverse_scaled_float64_v1"}
    identity = fingerprint(source)
    loaded = {s: split_to_arrays(s, data_dir=root, row_limit=row_limit) for s, _ in SPLITS}
    scaler = RobustScaler().fit(np.asarray(loaded["train"]["targets"], dtype=np.float32))
    values = {
        "label_robust_center": scaler.center_.astype(np.float32),
        "label_robust_scale": scaler.scale_.astype(np.float32),
        "flux_normalized_per_spectrum": np.asarray(normalize_flux),
        "target_names": np.asarray(TARGET_NAMES),
        "target_columns": np.asarray(TARGET_COLUMNS),
        "source_dataset": np.asarray(source["repo_id"] or source_kind),
        "dataset_id": np.asarray(identity),
        "wavelength_log10": np.linspace(GRID["log10_start"], GRID["log10_end"], GRID["points"]),
    }
    for split, short in SPLITS:
        x = np.asarray(loaded[split]["processed_flux"], dtype=np.float32)
        if normalize_flux:
            x = normalize_flux_per_spectrum(x)
        y = np.asarray(loaded[split]["targets"], dtype=np.float32)
        values.update({
            f"X_{short}_features": x,
            f"y_{short}_targets": scaler.transform(y).astype(np.float32),
            f"y_{short}_targets_original": y,
            f"{short}_ids": loaded[split]["metadata"]["spec_id"].to_numpy(dtype=np.int64),
        })
    validate_compact_dataset(values, root, row_limit=row_limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    metadata = {**source, "dataset_id": identity,
                "compact_sha256": sha256(temporary),
                "feature_sha256": {short: hashlib.sha256(memoryview(np.ascontiguousarray(values[f"X_{short}_features"])).cast("B")).hexdigest() for _, short in SPLITS}}
    temporary.replace(path)
    write_json(_manifest_path(path), metadata)
    print(f"Prepared {source_kind} cache: {path}", flush=True)
    return path


def load_compact(path, *, data_dir=None):
    from data_preprocessing.validation import validate_compact_dataset
    path = Path(path).resolve()
    root = Path(data_dir or path.parent)
    metadata_path = _manifest_path(path)
    if not metadata_path.exists():
        raise ValueError("Compact cache lacks provenance; prepare data in a fresh cache directory.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("grid") != GRID:
        raise ValueError("Compact wavelength grid differs from the benchmark.")
    if sha256(path) != metadata["compact_sha256"]:
        raise ValueError("Compact cache checksum/fingerprint changed.")
    if source_files(root) != metadata["split_files"]:
        raise ValueError("Compact source fingerprint changed.")
    identity_keys = ("schema_version", "source_kind", "repo_id", "revision", "split_files", "row_limit", "grid", "normalize_flux", "reference_labels")
    if fingerprint({k: metadata[k] for k in identity_keys}) != metadata["dataset_id"]:
        raise ValueError("Dataset provenance fingerprint changed.")
    data = np.load(path, allow_pickle=False)
    try:
        if str(data["dataset_id"].item()) != metadata["dataset_id"]:
            raise ValueError("Compact dataset identity differs from its manifest.")
        np.testing.assert_array_equal(data["target_names"], TARGET_NAMES)
        np.testing.assert_array_equal(data["target_columns"], TARGET_COLUMNS)
        np.testing.assert_array_equal(data["wavelength_log10"], np.linspace(GRID["log10_start"], GRID["log10_end"], GRID["points"]))
        validate_compact_dataset(data, root, row_limit=metadata["row_limit"])
    except BaseException:
        data.close()
        raise
    return data


def prepare_public(data_dir, *, row_limit=None, offline=False):
    from data_acquisition.hf_data import ensure_parquet_split, load_split_dataframe
    root = Path(data_dir).resolve()
    if row_limit is not None:
        if row_limit < 1:
            raise ValueError("row_limit must be positive.")
        root = root / "samples" / f"{REFERENCE['revision'][:12]}-n{row_limit}"
    root.mkdir(parents=True, exist_ok=True)
    for split, _ in SPLITS:
        path = root / f"{split}.parquet"
        if row_limit is None:
            ensure_parquet_split(split, data_dir=root, offline=offline)
        elif not path.exists():
            if offline:
                raise FileNotFoundError(f"Offline sample missing: {path}")
            # Load from the pinned public revision, with no private workspace dependency.
            frame = load_split_dataframe(split, row_limit=row_limit, data_dir=root)
            temporary = path.with_name(path.name + ".partial")
            frame.to_parquet(temporary, index=False)
            temporary.replace(path)
    return build_compact(root, source_kind="public" if row_limit is None else "public_sample", row_limit=row_limit)
