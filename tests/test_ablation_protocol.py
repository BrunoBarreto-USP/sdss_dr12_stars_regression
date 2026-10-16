"""Matched optimizer protocol and transactional ablation summaries."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np


class AblationProtocolTests(unittest.TestCase):
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
