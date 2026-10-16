"""Real short training round trip, with no historical files or network."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class StandaloneIntegrationTests(unittest.TestCase):
    def test_cnn_exports_predictions_and_json_with_missing_optional_diagnostics(self):
        from stellar_workflow.pipeline import make_fixture
        from stellar_workflow.datasets import load_compact
        from stellar_workflow.training import train_baselines
        from stellar_workflow.config import load_config
        from stellar_workflow.runs import Run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = make_fixture(root / "data")
            run = Run(root / "run")
            with load_compact(path) as data:
                train_baselines(run, load_config(epochs=1, batch_size=8, threads=2), data, models=["cnn"], pretrain_epochs=1)
            report = json.loads((run.path / "baselines/cnn/metrics.json").read_text(),
                                parse_constant=lambda value: self.fail("Nonstandard JSON number: " + value))
            self.assertTrue(np.isfinite(list(report["test_mae"].values())).all())
            self.assertTrue((run.path / "baselines/cnn/test_predictions.npz").exists())

    def test_offline_smoke_finishes_training_reports_and_figures(self):
        with tempfile.TemporaryDirectory(prefix="stellar integration ") as tmp:
            run = Path(tmp) / "run with spaces"
            env = os.environ.copy()
            env.update(HF_HUB_OFFLINE="1", TF_CPP_MIN_LOG_LEVEL="3", TF_ENABLE_ONEDNN_OPTS="0")
            result = subprocess.run(
                [sys.executable, "-m", "stellar_workflow", "smoke", "--offline",
                 "--run-dir", str(run), "--n-boot", "16"],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=180,
            )
            self.assertEqual(result.returncode, 0, result.stdout[-1000:] + result.stderr[-2000:])
            self.assertTrue((run / "retrained_paper_model.keras").exists())
            self.assertTrue((run / "history.csv").exists())
            with np.load(run / "test_predictions.npz") as data:
                self.assertEqual(data["y_pred"].shape, (8, 3))
                self.assertTrue(np.isfinite(data["y_pred"]).all())
            self.assertEqual(json.loads((run / "run_manifest.json").read_text())["stages"]["train"]["status"], "complete")
            self.assertTrue((run / "reports/evaluation.json").exists())
            self.assertTrue((run / "figures/manifest.json").exists())
