"""Remove repeated SDSS objects while preserving train/validation/test priority."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from astropy.io import fits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_acquisition.hf_data import CANONICAL_SPLITS, split_filename


IDENTIFIER_COLUMNS = ("BESTOBJID", "TARGETOBJID", "FLUXOBJID")
JOIN_COLUMNS = ("plate", "mjd", "fiber")


def _clean_identifier(value: object) -> str | None:
    identifier = str(value).strip()
    if identifier in {"", "0", "-1", "nan", "None"}:
        return None
    return identifier


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _load_metadata(data_dir: Path) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_parquet(
                data_dir / split_filename(split),
                columns=["spec_id", *JOIN_COLUMNS],
            ).assign(split=split)
            for split in CANONICAL_SPLITS
        ],
        ignore_index=True,
    )


def _load_object_identifiers(fits_path: Path) -> pd.DataFrame:
    with fits.open(fits_path, memmap=True) as hdul:
        table = hdul[1].data
        return pd.DataFrame(
            {
                "plate": np.asarray(table["PLATE"]),
                "mjd": np.asarray(table["MJD"]),
                "fiber": np.asarray(table["FIBER"]),
                "bestobjid": np.asarray(table["BESTOBJID"]),
                "targetobjid": np.asarray(table["TARGETOBJID"]),
                "fluxobjid": np.asarray(table["FLUXOBJID"]),
            }
        )


def _select_spectra(data_dir: Path, fits_path: Path) -> tuple[dict[str, set[int]], dict[str, int]]:
    metadata = _load_metadata(data_dir)
    identifiers = _load_object_identifiers(fits_path)
    for column in JOIN_COLUMNS:
        metadata[column] = pd.to_numeric(metadata[column]).astype("int64")
        identifiers[column] = pd.to_numeric(identifiers[column]).astype("int64")

    rows = metadata.merge(
        identifiers,
        on=list(JOIN_COLUMNS),
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not rows["_merge"].eq("both").all():
        missing = int((rows["_merge"] != "both").sum())
        raise RuntimeError(f"{missing} spectra did not match an SSPP FITS row.")

    aliases = [
        [_clean_identifier(getattr(row, column)) for column in ("bestobjid", "targetobjid", "fluxobjid")]
        for row in rows.itertuples(index=False)
    ]
    groups = _DisjointSet()
    for row_aliases in aliases:
        valid_aliases = [identifier for identifier in row_aliases if identifier is not None]
        for identifier in valid_aliases[1:]:
            groups.union(valid_aliases[0], identifier)

    rows["object_group"] = [
        groups.find(next(identifier for identifier in row_aliases if identifier is not None))
        if any(row_aliases)
        else None
        for row_aliases in aliases
    ]
    rows["split_priority"] = rows["split"].map({split: index for index, split in enumerate(CANONICAL_SPLITS)})
    valid_rows = rows.dropna(subset=["object_group"]).sort_values(
        ["split_priority", "spec_id"],
        kind="stable",
    )
    kept_rows = valid_rows.drop_duplicates("object_group", keep="first")
    selected = {
        split: set(kept_rows.loc[kept_rows["split"] == split, "spec_id"].astype(int))
        for split in CANONICAL_SPLITS
    }
    summary = {
        "input_rows": len(rows),
        "removed_without_object_id": int(rows["object_group"].isna().sum()),
        "removed_repeated_object": len(valid_rows) - len(kept_rows),
        "remaining_rows": len(kept_rows),
        **{f"remaining_{split}": len(selected[split]) for split in CANONICAL_SPLITS},
    }
    return selected, summary


def _write_filtered_parquet(source: Path, destination: Path, selected_ids: set[int]) -> int:
    source_file = pq.ParquetFile(source)
    spec_id_index = source_file.schema_arrow.get_field_index("spec_id")
    values = pa.array(sorted(selected_ids), type=pa.int64())
    written = 0
    writer: pq.ParquetWriter | None = None
    try:
        for batch in source_file.iter_batches(batch_size=4096):
            keep = pc.is_in(batch.column(spec_id_index), value_set=values)
            filtered = batch.filter(keep)
            if not filtered.num_rows:
                continue
            if writer is None:
                writer = pq.ParquetWriter(destination, filtered.schema, compression="snappy")
            writer.write_batch(filtered)
            written += filtered.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError(f"No selected rows were written to {destination}.")
    return written


def deduplicate_splits(*, data_dir: Path, fits_path: Path, in_place: bool) -> dict[str, int]:
    data_dir = data_dir.resolve()
    fits_path = fits_path.resolve()
    selected, summary = _select_spectra(data_dir, fits_path)
    staging_dir = data_dir / ".object_deduplication_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()
    try:
        for split in CANONICAL_SPLITS:
            source = data_dir / split_filename(split)
            destination = staging_dir / source.name
            written = _write_filtered_parquet(source, destination, selected[split])
            if written != len(selected[split]):
                raise RuntimeError(f"Expected {len(selected[split])} {split} rows, wrote {written}.")

        if in_place:
            for split in CANONICAL_SPLITS:
                source = data_dir / split_filename(split)
                staged = staging_dir / source.name
                os.replace(staged, source)
        else:
            print(f"Filtered files are staged at: {staging_dir}")
            return summary
    finally:
        if in_place and staging_dir.exists():
            shutil.rmtree(staging_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--fits", type=Path, required=True, help="Path to ssppOut-dr12.fits.")
    parser.add_argument("--in-place", action="store_true", help="Replace the local split Parquet files after validation.")
    args = parser.parse_args()
    summary = deduplicate_splits(data_dir=args.data_dir, fits_path=args.fits, in_place=args.in_place)
    print("Object-level deduplication summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
