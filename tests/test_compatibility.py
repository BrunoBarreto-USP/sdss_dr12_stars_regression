"""Legacy entry points remain thin adapters to the same public workflow."""
import subprocess
import sys
import unittest
from unittest.mock import patch


class CompatibilityTests(unittest.TestCase):
    def test_legacy_output_alias(self):
        from stellar_workflow.compat import legacy_main
        with patch("stellar_workflow.cli.main") as entry:
            legacy_main("train", ["--out-dir", "folder with spaces", "--epochs", "1"])
        entry.assert_called_once_with(["train", "--run-dir", "folder with spaces", "--epochs", "1"])

    def test_documented_wrappers_have_workflow_help(self):
        for name in ("retrain_paper_with_history", "finalize_corrected_experiment", "audit_research_pipeline"):
            result = subprocess.run([sys.executable, "scripts/" + name + ".py", "--help"],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--data-dir", result.stdout)


if __name__ == "__main__":
    unittest.main()
