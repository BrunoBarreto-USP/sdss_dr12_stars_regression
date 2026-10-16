"""Standalone execution contracts, using local synthetic data only."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd


class WorkflowContracts(unittest.TestCase):
    def test_reference_labels_keep_float64_inverse_scaling(self):
        from stellar_workflow.predictions import physical_labels
        data = {
            "y_test_targets": np.array([[0.125, -0.5, 0.25]], dtype=np.float32),
            "label_robust_center": np.array([5000, -1, 3], dtype=np.float32),
            "label_robust_scale": np.array([1200, 0.5, 2], dtype=np.float32),
        }
        np.testing.assert_array_equal(physical_labels(data, "test"), [[5150, -1.25, 3.5]])
        self.assertEqual(physical_labels(data, "test").dtype, np.float64)

    def test_predictions_reject_different_object_order(self):
        from stellar_workflow.predictions import write_predictions, read_predictions, assert_aligned
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.npz", Path(tmp) / "b.npz"
            truth = np.array([[5000., -1., 3.], [6000., 0., 4.]])
            write_predictions(a, [1, 2], truth, truth, dataset_id="fixture")
            write_predictions(b, [2, 1], truth, truth, dataset_id="fixture")
            with self.assertRaisesRegex(ValueError, "IDs|identif|order"):
                assert_aligned(read_predictions(a), read_predictions(b))

    def test_stage_append_and_collision_are_separate(self):
        from stellar_workflow.runs import Run
        with tempfile.TemporaryDirectory(prefix="stellar path ") as tmp:
            run = Run(Path(tmp) / "run")
            with run.stage("train", ["model.keras"]):
                (run.path / "model.keras").write_bytes(b"checkpoint")
            with run.stage("baseline:cnn", ["baselines/cnn/model.keras"]):
                p = run.path / "baselines/cnn/model.keras"
                p.parent.mkdir(parents=True)
                p.write_bytes(b"cnn")
            with self.assertRaises(FileExistsError):
                with run.stage("train", ["model.keras"]):
                    self.fail("Protected stage must never start.")
            self.assertEqual((run.path / "model.keras").read_bytes(), b"checkpoint")
            self.assertEqual(json.loads((run.path / "run_manifest.json").read_text())["stages"]["train"]["status"], "complete")

    def test_derived_replacement_cannot_replace_training_directory(self):
        from stellar_workflow.runs import derived_output
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "history.csv").write_text("epoch,loss\n0,1\n")
            with self.assertRaises(ValueError):
                with derived_output(root, replace=True):
                    self.fail("Training directory must remain protected.")
            self.assertTrue((root / "history.csv").exists())

    def test_derived_replacement_preserves_linear_baseline_checkpoint(self):
        from stellar_workflow.runs import derived_output
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.joblib").write_bytes(b"linear model")
            with self.assertRaises(ValueError):
                with derived_output(root, replace=True):
                    pass
            self.assertEqual((root / "model.joblib").read_bytes(), b"linear model")

    def test_tuning_archive_selects_original_winner(self):
        from stellar_workflow.config import load_tuning_history
        trials = load_tuning_history()
        completed = [t for t in trials if t["status"] == "COMPLETED"]
        self.assertEqual(len(trials), 100)
        self.assertEqual(min(completed, key=lambda t: t["score"])["trial_id"], "trial_099")

    def test_tuning_archive_retains_original_json_bytes(self):
        from stellar_workflow.config import ROOT
        import hashlib
        archive = json.loads((ROOT / "reference/tuning_history.json").read_text())
        self.assertEqual(len(archive["original_json"]), 100)
        for name, raw in archive["original_json"].items():
            self.assertEqual(hashlib.sha256(raw.encode("utf-8")).hexdigest(), archive["original_sha256"][name])

    def test_cache_rejects_missing_features(self):
        from data_preprocessing.validation import validate_compact_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = np.array([[5000, -1, 3], [6000, 0, 4]], dtype=np.float32)
            data = {"label_robust_center": [5500, -.5, 3.5], "label_robust_scale": [500, .5, .5]}
            for offset, (split, short) in enumerate((("train", "train"), ("validation", "val"), ("test", "test"))):
                ids = np.array([offset * 2, offset * 2 + 1])
                pd.DataFrame({"spec_id": ids, "catalog_teff": labels[:, 0], "catalog_feh": labels[:, 1], "catalog_logg": labels[:, 2]}).to_parquet(root / f"{split}.parquet")
                data[f"{short}_ids"] = ids
                data[f"y_{short}_targets"] = np.array([[-1, -1, -1], [1, 1, 1]], dtype=np.float32)
            with self.assertRaisesRegex(ValueError, "feature"):
                validate_compact_dataset(data, root)

    def test_cli_help_does_not_import_torch(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-c", "import sys; from stellar_workflow.cli import parser; p=parser(); assert 'torch' not in sys.modules; assert 'tensorflow' not in sys.modules; p.parse_args(['--help'])"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reproduce-paper", result.stdout)

    def test_explicit_validation_truth_is_used_when_test_has_same_length(self):
        from training.train_best_tuner_trial import _physical_mae
        class Model:
            def predict(self, x, **kwargs):
                return [np.zeros((len(x), 1)) for _ in range(3)]
        data = {"label_robust_center": np.array([5000., -1., 3.]),
                "label_robust_scale": np.ones(3),
                "y_test_targets_original": np.full((2, 3), 9000.)}
        values = _physical_mae(Model(), np.zeros((2, 4000)), np.zeros((2, 3)), data,
                              batch_size=2, true_targets=np.array([[5001., -.75, 3.5]] * 2))
        self.assertEqual(values, {"Teff": 1., "[Fe/H]": .25, "log g": .5})

    def test_cache_rejects_changed_input_with_identical_ids_and_labels(self):
        from data_preprocessing.validation import validate_compact_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = np.array([[5000, -1, 3], [6000, 0, 4]], dtype=np.float32)
            data = {"label_robust_center": np.array([5500, -.5, 3.5]), "label_robust_scale": np.array([500, .5, .5])}
            for offset, (split, short) in enumerate((("train", "train"), ("validation", "val"), ("test", "test"))):
                ids = np.array([offset * 2, offset * 2 + 1])
                features = np.zeros((2, 4000), dtype=np.float32)
                pd.DataFrame({"spec_id": ids, "catalog_teff": labels[:, 0], "catalog_feh": labels[:, 1], "catalog_logg": labels[:, 2], "processed_flux": list(features)}).to_parquet(root / f"{split}.parquet")
                data[f"{short}_ids"] = ids
                data[f"X_{short}_features"] = features
                data[f"y_{short}_targets"] = np.array([[-1, -1, -1], [1, 1, 1]], dtype=np.float32)
            validate_compact_dataset(data, root)
            data["X_test_features"][0, 2000] = 1
            with self.assertRaisesRegex(ValueError, "feature"):
                validate_compact_dataset(data, root)

    def test_baseline_collision_is_checked_before_any_other_fit(self):
        from stellar_workflow.training import train_baselines
        from stellar_workflow.config import load_config
        from stellar_workflow.runs import Run
        with tempfile.TemporaryDirectory() as tmp:
            run = Run(tmp)
            p = run.path / "baselines/cnn/fabbro_cnn.keras"
            p.parent.mkdir(parents=True)
            p.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                train_baselines(run, load_config(), {}, models=("dnn", "cnn"))
            self.assertFalse((run.path / "baselines/dnn").exists())
            self.assertEqual(p.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
