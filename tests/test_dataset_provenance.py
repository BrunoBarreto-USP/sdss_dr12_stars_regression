"""Cache integrity and atomic verified download behavior."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd


def local_splits(root):
    for offset, split in enumerate(("train", "validation", "test")):
        x = np.arange(12000, dtype=np.float32).reshape(3, 4000) / 100
        pd.DataFrame({
            "spec_id": np.arange(offset * 3, offset * 3 + 3),
            "plate": 1, "mjd": 1, "fiber": [1, 2, 3], "ra": [1., 2., 3.],
            "dec": [0., 1., 2.], "snr": [20., 40., 60.], "rv_adop": 0.,
            "catalog_teff": [4000., 5000., 6000.], "catalog_feh": [-2., -1., 0.],
            "catalog_logg": [1., 2., 3.], "processed_flux": list(x),
        }).to_parquet(root / f"{split}.parquet")


class DatasetProvenanceTests(unittest.TestCase):
    def test_failed_sample_stream_does_not_silently_download_full_dataset(self):
        from data_acquisition.hf_data import load_split_dataframe
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp, patch("datasets.load_dataset", side_effect=OSError("network failure")), patch(
                "data_acquisition.hf_data.ensure_parquet_split", side_effect=AssertionError("Unexpected full download")):
            with self.assertRaisesRegex(RuntimeError, "sample|stream"):
                load_split_dataframe("train", row_limit=3, data_dir=tmp)

    def test_public_sample_keeps_requested_limit_for_portable_recovery(self):
        from stellar_workflow.datasets import prepare_public, REFERENCE
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "samples" / (REFERENCE["revision"][:12] + "-n3")
            sample.mkdir(parents=True)
            local_splits(sample)
            path = prepare_public(root, row_limit=3, offline=True)
            report = json.loads((path.parent / "compact_manifest.json").read_text())
            self.assertEqual(report["row_limit"], 3)

    def test_cache_fingerprint_detects_changed_archive(self):
        from stellar_workflow.datasets import build_compact, load_compact
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_splits(root)
            path = build_compact(root, source_kind="fixture")
            with load_compact(path) as data:
                self.assertEqual(data["X_test_features"].shape, (3, 4000))
            with path.open("ab") as stream:
                stream.write(b"unexpected")
            with self.assertRaisesRegex(ValueError, "fingerprint|checksum"):
                load_compact(path)

    def test_cache_rejects_changed_source_and_grid(self):
        from stellar_workflow.datasets import build_compact, load_compact
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_splits(root)
            path = build_compact(root, source_kind="fixture")
            manifest = root / "compact_manifest.json"
            original = manifest.read_text()
            values = json.loads(original)
            values["grid"]["log10_end"] = 4.
            manifest.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, "grid"):
                load_compact(path)
            manifest.write_text(original)
            frame = pd.read_parquet(root / "test.parquet")
            frame.loc[0, "catalog_teff"] += 1
            frame.to_parquet(root / "test.parquet")
            with self.assertRaisesRegex(ValueError, "source|fingerprint"):
                load_compact(path)

    def test_offline_missing_download_leaves_no_cache(self):
        from stellar_workflow.datasets import download_verified
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.parquet"
            with self.assertRaises(FileNotFoundError):
                download_verified("https://example.invalid/no-network", path, "0" * 64, offline=True)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
