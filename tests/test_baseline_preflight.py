"""Dependency errors must point to usable installation instructions."""
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch


class BaselinePreflightTests(unittest.TestCase):
    def test_missing_torch_names_existing_requirements_before_creating_outputs(self):
        from stellar_workflow.config import ROOT, load_config
        from stellar_workflow.runs import Run
        from stellar_workflow.training import train_baselines
        with tempfile.TemporaryDirectory() as tmp:
            run = Run(Path(tmp) / "run")
            with patch.dict("sys.modules", {"torch": None}):
                with self.assertRaises(ImportError) as error:
                    train_baselines(run, load_config(), {}, models=["transformer"], allow_cpu=True)
            requirement = re.search(r"requirements[\w/.-]*\.txt", str(error.exception))
            self.assertIsNotNone(requirement, "The error must identify the optional dependency file.")
            self.assertTrue((ROOT / requirement.group()).is_file(), "The suggested installation file must exist.")
            self.assertFalse((run.path / "baselines").exists())
            self.assertEqual(run.read()["stages"], {})


if __name__ == "__main__":
    unittest.main()
