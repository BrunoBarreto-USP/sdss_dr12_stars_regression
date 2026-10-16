"""Optional CPU checks for Transformer selection; no network or GPU required."""
import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "Install optional Transformer requirements.")
class TransformerSelectionTests(unittest.TestCase):
    def test_validation_loss_is_invariant_to_partial_batch_partition(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from finetuning.train_spectral_transformer import _evaluate
        torch.set_num_threads(2)
        inputs = torch.ones((4846, 3))
        inputs[4096:] = 10
        dataset = TensorDataset(inputs, torch.zeros_like(inputs))
        results = []
        for batch_size in (1024, 999, 4846):
            metrics = _evaluate(
                torch.nn.Identity(), DataLoader(dataset, batch_size=batch_size),
                device=torch.device("cpu"), weights=torch.ones(3),
                centers=np.zeros(3), scales=np.ones(3), amp=False,
            )
            results.append(metrics.loss)
        for loss in results:
            self.assertAlmostEqual(loss, 9173 / 4846, places=6)
@unittest.skipUnless(importlib.util.find_spec("torch"), "Install optional Transformer requirements.")
class TransformerExportGuards(unittest.TestCase):
    def test_tiny_transformer_exports_portable_aligned_run(self):
        import tempfile
        import json
        from pathlib import Path
        from stellar_workflow.pipeline import make_fixture
        from stellar_workflow.predictions import read_predictions, physical_labels
        from stellar_workflow.datasets import load_compact
        from finetuning.train_spectral_transformer import parse_args, train_spectral_transformer
        import torch
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compact = make_fixture(root / "data")
            folder = root / "transformer"
            args = parse_args(["--data", str(compact), "--output-dir", str(folder), "--allow-cpu",
                               "--epochs", "1", "--batch-size", "8", "--workers", "0",
                               "--d-model", "16", "--layers", "1", "--heads", "2"])
            train_spectral_transformer(args)
            saved = read_predictions(folder / "test_predictions.npz")
            with load_compact(compact) as data:
                np.testing.assert_array_equal(saved["y_true"], physical_labels(data))
                self.assertEqual(saved["dataset_id"], str(data["dataset_id"].item()))
            metrics = json.loads((folder / "test_metrics.json").read_text())
            self.assertEqual(metrics["selection_metric_version"], "object_weighted_huber_v2")
            self.assertEqual(len(json.loads((folder / "history.json").read_text())), 1)

    def test_existing_checkpoint_is_preserved_before_training(self):
        import tempfile
        from pathlib import Path
        from finetuning.train_spectral_transformer import parse_args, train_spectral_transformer
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "best_model.pt").write_bytes(b"original checkpoint")
            args = parse_args(["--output-dir", str(folder), "--data", str(folder / "missing.npz"), "--allow-cpu"])
            with self.assertRaises(FileExistsError):
                train_spectral_transformer(args)
            self.assertEqual((folder / "best_model.pt").read_bytes(), b"original checkpoint")
