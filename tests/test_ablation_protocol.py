"""Matched optimizer protocol and transactional ablation summaries."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np


class AblationProtocolTests(unittest.TestCase):
    def make_prediction_run(self, root, *, single_dataset_id="fixture", single_targets=None):
        from stellar_workflow.config import TARGET_NAMES
        from stellar_workflow.predictions import write_predictions
        from stellar_workflow.runs import Run
        run = Run(root / "run")
        # Equal columns ensure target-name mistakes cannot be caught by labels alone.
        truth = np.array([[1., 1., 1.], [2., 2., 2.]])
        write_predictions(run.path / "test_predictions.npz", [1, 2], truth, truth, dataset_id="fixture")
        for name in ("no_noise_augmentation", "no_skip", "no_layernorm", "no_residual_no_layernorm"):
            folder = run.path / "ablations" / name
            write_predictions(folder / "test_predictions.npz", [1, 2], truth, truth, dataset_id="fixture")
            (folder / "metrics.json").write_text(json.dumps({"params": 1}))
        for i, name in enumerate(("single_teff", "single_feh", "single_logg")):
            folder = run.path / "ablations" / name
            target = (single_targets or TARGET_NAMES)[i]
            write_predictions(folder / "test_predictions.npz", [1, 2], truth[:, i:i+1], truth[:, i:i+1],
                              dataset_id=single_dataset_id, target_names=(target,))
            (folder / "metrics.json").write_text(json.dumps({"params": 1}))
        return run

    def test_single_target_comparison_rejects_other_dataset_before_reporting(self):
        from stellar_workflow.evaluation import ablation_reports
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_prediction_run(Path(tmp), single_dataset_id="other-features")
            with self.assertRaisesRegex(ValueError, "fingerprints"):
                ablation_reports(run, n_boot=4)
            self.assertFalse((run.path / "reports").exists())

    def test_single_target_comparison_rejects_wrong_target_with_identical_labels(self):
        from stellar_workflow.evaluation import ablation_reports
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_prediction_run(Path(tmp), single_targets=("[Fe/H]", "Teff", "log g"))
            with self.assertRaisesRegex(ValueError, "target order"):
                ablation_reports(run, n_boot=4)
            self.assertFalse((run.path / "reports").exists())

    def test_single_target_comparison_accepts_aligned_and_legacy_archives(self):
        from stellar_workflow.evaluation import ablation_reports
        with tempfile.TemporaryDirectory() as tmp:
            for fingerprint in ("fixture", ""):
                with self.subTest(fingerprint=fingerprint):
                    root = Path(tmp) / (fingerprint or "legacy")
                    run = self.make_prediction_run(root, single_dataset_id=fingerprint)
                    rows = ablation_reports(run, n_boot=4)
                    self.assertEqual(len(rows), 15)
                    singles = [row for row in rows if row["variant"] == "no_multitask"]
                    self.assertEqual(len(singles), 3)
                    self.assertTrue(all(row["params"] == 3 and row["mae_difference"] == 0 for row in singles))
                    self.assertTrue((run.path / "reports/ablations/ablation_summary.csv").is_file())

    def test_all_ablation_learning_rates_match_main_for_small_batches(self):
        from stellar_workflow.pipeline import make_fixture
        from stellar_workflow.datasets import load_compact
        from stellar_workflow.config import load_config
        from stellar_workflow.training import fit_model, runtime
        from scripts.retrain_corrected_ablations import VARIANTS
        config = load_config(epochs=1, batch_size=8, threads=2)
        tf = runtime(config)
        rates = []
        class BeforeFit(Exception):
            pass
        def observe(model, *args, **kwargs):
            rates.append(model.optimizer._learning_rate.initial_learning_rate)
            raise BeforeFit
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compact = make_fixture(root / "data")
            with load_compact(compact) as data, patch.object(tf.keras.Model, "fit", observe):
                for variant in (None, *VARIANTS):
                    with self.assertRaises(BeforeFit):
                        fit_model(config, data, root / str(variant), variant=variant)
                    tf.keras.backend.clear_session()
        np.testing.assert_allclose(rates, [rates[0]] * len(rates), rtol=1e-12)

    def test_failed_ablation_summary_leaves_no_partial_derived_report(self):
        from stellar_workflow.evaluation import ablation_reports
        from stellar_workflow.runs import Run
        from stellar_workflow.config import TARGET_NAMES
        from scripts.retrain_corrected_ablations import VARIANTS
        with tempfile.TemporaryDirectory() as tmp:
            run = Run(Path(tmp) / "run")
            truth = np.array([[5000., -1., 3.], [6000., 0., 4.]])
            def read(path):
                index = {"single_teff":0,"single_feh":1,"single_logg":2}.get(Path(path).parent.name)
                indices = [index] if index is not None else [0,1,2]
                return {"test_ids":np.array([1,2]),"target_names":tuple(TARGET_NAMES[i] for i in indices),
                        "dataset_id":"fixture","y_true":truth[:,indices],"y_pred":truth[:,indices]}
            for variant in VARIANTS:
                folder = run.path / "ablations" / variant
                folder.mkdir(parents=True)
                (folder / "metrics.json").write_text(json.dumps({"params":1}))
            output = run.path / "derived"
            with patch("stellar_workflow.evaluation.read_predictions", side_effect=read), patch(
                    "stellar_workflow.evaluation.write_json", side_effect=OSError("simulated report failure")):
                with self.assertRaises(OSError):
                    ablation_reports(run, n_boot=4, output_dir=output)
            self.assertFalse(output.exists() and any(output.iterdir()), "Partial reports must not become committed output.")


if __name__ == "__main__":
    unittest.main()
