"""Execute the notebook launcher without running its expensive optional stages."""
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def notebook_workflow():
    notebook = json.loads((ROOT / "orchestration.ipynb").read_text(encoding="utf-8"))
    setup = ast.parse("".join(notebook["cells"][1]["source"]))
    launcher = next(node for node in setup.body if isinstance(node, ast.FunctionDef) and node.name == "workflow")
    namespace = dict(os=os, subprocess=subprocess, sys=sys, ROOT=ROOT)
    exec(compile(ast.Module(body=[launcher], type_ignores=[]), "orchestration.ipynb", "exec"), namespace)
    return namespace["workflow"]


class NotebookWorkflowTests(unittest.TestCase):
    def test_default_launcher_does_not_limit_long_opt_in_stages(self):
        workflow = notebook_workflow()
        with patch.object(subprocess, "run", wraps=subprocess.run) as launch, redirect_stdout(io.StringIO()):
            workflow("--help")
        self.assertIsNone(launch.call_args.kwargs.get("timeout"), "Full experiments must not stop after one hour.")

    def test_explicit_launcher_timeout_is_enforced(self):
        import inspect
        workflow = notebook_workflow()
        self.assertIn("timeout", inspect.signature(workflow).parameters)
        with self.assertRaises(subprocess.TimeoutExpired):
            workflow("--help", timeout=0)


if __name__ == "__main__":
    unittest.main()
