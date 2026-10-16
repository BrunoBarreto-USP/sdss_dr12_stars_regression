"""Regression checks for target alignment, cache reuse and the selected architecture."""

from pathlib import Path
import json
import os
import sys
import tempfile
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)

from data_acquisition.hf_data import load_split_dataframe
from data_preprocessing.augmentation import augment_training_data
from data_preprocessing.validation import validate_compact_dataset, TARGET_COLUMNS
from results_and_evaluations.prediction_utils import prediction_matrix, OUTPUT_NAMES
from results_and_evaluations.evaluation import _extract_sigma
from results_and_evaluations.paired_bootstrap_all_models import (
    bootstrap_paired_differences,
)
from training.training import (
    compute_inverse_variance_loss_weights,
    NormalizedMaeSumCallback,
    _create_dataset,
)
from training.train_paper_hyperparams import build_paper_fixed_model, FixedPaperHP
from training.run_paper_ablations import build_paper_ablation_model
from model_definitions.model_definitions import WarmupCosineDecay
from retrain_corrected_ablations import (
    cached_augmentation,
    indexed_dataset,
    build_variant,
)
from analyze_parameter_ranges import summarize


class ResearchPipelineTests(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)

    def test_named_predictions_preserve_canonical_target_order(self):
        raw = {
            "logg_output": np.array([3, 6]),
            "teff_output": np.array([1, 4]),
            "feh_output": np.array([2, 5]),
        }
        np.testing.assert_array_equal(prediction_matrix(raw), [[1, 2, 3], [4, 5, 6]])

    def test_invalid_predictions_are_rejected(self):
        for raw in ({"teff_output": [1]}, np.array([[1, np.nan, 3]]), [np.ones(2)]):
            with self.assertRaises(ValueError):
                prediction_matrix(raw)

    def test_paired_bootstrap_keeps_zero_and_signed_constant_effects(self):
        values = np.zeros((40, 3, 2))
        values[:, 0, 0] = 2
        values[:, 2, 1] = -3
        means, ci = bootstrap_paired_differences(values, n_boot=100, seed=42)
        np.testing.assert_array_equal(ci[0], means)
        np.testing.assert_array_equal(ci[1], means)
        self.assertEqual(means[0, 0], 2)
        self.assertEqual(means[2, 1], -3)

    def test_historical_uncertainty_softplus_does_not_overflow(self):
        sigma = _extract_sigma([np.array([[0, 1000.0], [0, -1000.0]])])
        self.assertTrue(np.isfinite(sigma).all())
        self.assertGreater(sigma[0, 0], 30)

    def test_augmentation_without_new_copies_keeps_labels_aligned(self):
        x = np.arange(42, dtype=np.float32).reshape(7, 6)
        y = {"target": x[:, 0].copy()}
        features, labels = augment_training_data(x, y, aug_factor=1, seed=3)
        np.testing.assert_array_equal(features[:, 0], labels["target"])
        self.assertEqual(len(features), 7)

    def test_invalid_augmentation_arguments_are_rejected(self):
        x = np.ones((5, 3), np.float32)
        y = {"target": np.arange(5, dtype=np.float32)}
        for kwargs in ({"aug_factor": 0}, {"aug_factor": 2, "noise_level": 0.005}):
            with self.assertRaises(ValueError):
                augment_training_data(x, y, **kwargs)
        with self.assertRaises(ValueError):
            augment_training_data(x, {"target": np.ones(4)})

    def test_disk_backed_copies_match_original_augmentation_exactly(self):
        x = np.random.default_rng(5).normal(size=(519, 27)).astype(np.float32)
        y = {"target": np.arange(len(x), dtype=np.float32)}
        expected, labels = augment_training_data(
            x, y, aug_factor=3, noise_level=0.05, seed=42
        )
        with tempfile.TemporaryDirectory() as folder:
            cached, order = cached_augmentation(x, Path(folder) / "copies.npy")
            np.testing.assert_array_equal(expected, cached[order])
            np.testing.assert_array_equal(
                labels["target"], np.tile(y["target"], 3)[order]
            )
            del cached

    def test_disk_cache_rejects_changed_training_inputs(self):
        x = np.arange(30, dtype=np.float32).reshape(10, 3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "copies.npy"
            cached, _ = cached_augmentation(x, path)
            del cached
            changed = x.copy()
            changed[0, 0] += 1
            with self.assertRaises(ValueError):
                cached_augmentation(changed, path)

    def test_no_noise_control_keeps_feature_label_alignment(self):
        x = np.arange(300, dtype=np.float32).reshape(100, 3)
        labels = {OUTPUT_NAMES[0]: x[:, 0]}
        order = np.random.default_rng(8).permutation(300)
        dataset = indexed_dataset(
            x, labels, order, [OUTPUT_NAMES[0]], repeat_original=True
        )
        features, target = next(iter(dataset))
        np.testing.assert_array_equal(
            features.numpy()[:, 0], target[OUTPUT_NAMES[0]].numpy()
        )

    def test_inverse_variance_weights_are_normalized(self):
        y = {"a": np.array([-1.0, 1.0]), "b": np.array([-2.0, 2.0])}
        weights = compute_inverse_variance_loss_weights(y, ("a", "b"))
        self.assertAlmostEqual(sum(weights.values()), 2)
        self.assertAlmostEqual(weights["a"] / weights["b"], 4)

    def test_constant_target_weighting_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_inverse_variance_loss_weights({"a": np.ones(3)}, ("a",))

    def test_archived_trial_selection_uses_recorded_training_scales(self):
        run = ROOT / "outputs/paper_figures/single_column/corrected_best_trial"
        config = json.loads((run / "run_config.json").read_text())
        trial = json.loads(
            (
                ROOT
                / "results_and_evaluations/keras_tuner/stellar_baseline_orchestration/trial_099/trial.json"
            ).read_text()
        )
        values = {
            k: v["observations"][0]["value"][0]
            for k, v in trial["metrics"]["metrics"].items()
        }
        expected = sum(
            values["val_" + k + "_mae"] / config["target_standard_deviations"][k]
            for k in OUTPUT_NAMES
        )
        self.assertAlmostEqual(expected, trial["score"], places=12)

    def test_validation_metric_uses_all_task_scales(self):
        callback = NormalizedMaeSumCallback(
            dict(zip(OUTPUT_NAMES, (2.0, 4.0, 5.0))), OUTPUT_NAMES
        )
        logs = {f"val_{k}_mae": v for k, v in zip(OUTPUT_NAMES, (1.0, 2.0, 5.0))}
        callback.on_epoch_end(0, logs)
        self.assertEqual(logs["val_normalized_mae_sum"], 2.0)

    def test_validation_keeps_the_last_partial_batch(self):
        x = np.zeros((7, 3), np.float32)
        y = {"a": np.arange(7, dtype=np.float32)}
        validation = _create_dataset(x, y, 4, shuffle=False)
        self.assertEqual(sum(len(batch[0]) for batch in validation), 7)

    def test_selected_and_legacy_ablation_builders_agree(self):
        model = build_paper_fixed_model(
            input_dim=4000,
            total_steps=80,
            loss_weights=dict.fromkeys(OUTPUT_NAMES, 1.0),
            batch_size=256,
            jit_compile=False,
        )
        self.assertEqual(model.count_params(), 530275)
        for name, widths in zip(OUTPUT_NAMES, ((16,), (48, 32, 16), (48, 32, 16))):
            self.assertEqual(
                tuple(
                    l.units
                    for l in model.layers
                    if l.name.startswith(name + "_head_dense_")
                ),
                widths,
            )
        legacy = build_paper_ablation_model(
            input_dim=4000,
            total_steps=80,
            batch_size=256,
            loss_weights=dict.fromkeys(OUTPUT_NAMES, 1.0),
            use_residual=True,
            use_layernorm=True,
        )
        self.assertEqual(legacy.count_params(), 530275)

    def test_separate_ablation_graphs_remove_only_requested_components(self):
        for name, params, skip, norm in (
            ("no_skip", 525891, False, True),
            ("no_layernorm", 529379, True, False),
            ("no_noise_augmentation", 530275, True, True),
        ):
            model, _ = build_variant(name, 80, 344, dict.fromkeys(OUTPUT_NAMES, 1.0))
            self.assertEqual(model.count_params(), params)
            self.assertEqual(
                any(isinstance(l, tf.keras.layers.Add) for l in model.layers), skip
            )
            self.assertEqual(
                any(
                    isinstance(l, tf.keras.layers.LayerNormalization)
                    for l in model.layers
                ),
                norm,
            )

    def test_warmup_and_cosine_decay_are_continuous_and_monotonic_after_peak(self):
        schedule = WarmupCosineDecay(0.005, total_steps=100, warmup_steps=5, alpha=0.05)
        values = np.array([float(schedule(i)) for i in range(101)])
        self.assertTrue(np.all(np.diff(values[:6]) >= 0))
        self.assertTrue(np.all(np.diff(values[5:]) <= 0))
        self.assertAlmostEqual(values[-1], 0.00025, places=8)

    def test_parameter_bins_include_final_right_edge(self):
        true = np.array([[4000, -4.5, 0], [5000, -1, 4], [9500, 0.75, 5.0]])
        rows = summarize(true, true + np.array([2.0, 0.1, -0.2]), n_boot=32, seed=42)
        for target in ("Teff", "[Fe/H]", "log g"):
            self.assertEqual(int(rows[rows.target == target].n.sum()), 3)

    def test_missing_local_columns_raise_key_error_with_filename(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "train.parquet").touch()
            with patch(
                "data_acquisition.hf_data._read_parquet",
                return_value=pd.DataFrame({"spec_id": [1]}),
            ):
                with self.assertRaisesRegex(KeyError, "train.parquet"):
                    load_split_dataframe(
                        "train", columns=["spec_id", "absent"], data_dir=folder
                    )

    def test_compact_cache_rejects_changed_labels_or_scaler(self):
        data = {}
        with tempfile.TemporaryDirectory() as folder:
            for number, (split, short) in enumerate(
                (("train", "train"), ("validation", "val"), ("test", "test"))
            ):
                labels = np.array(
                    [[4000, -2, 1], [5000, -1, 2], [6000, 0, 3]], dtype=np.float32
                )
                ids = np.arange(number * 3, number * 3 + 3)
                frame = pd.DataFrame(labels, columns=TARGET_COLUMNS)
                frame["spec_id"] = ids
                frame.to_parquet(Path(folder) / f"{split}.parquet")
                data[f"{short}_ids"] = ids
                data[f"y_{short}_targets"] = (labels - [5000, -1, 2]) / [1000, 1, 1]
            data["label_robust_center"] = np.array([5000, -1, 2])
            data["label_robust_scale"] = np.array([1000, 1, 1])
            self.assertEqual(validate_compact_dataset(data, Path(folder))["test"], 3)
            data["y_test_targets"][0, 0] += 1
            with self.assertRaises(AssertionError):
                validate_compact_dataset(data, Path(folder))


if __name__ == "__main__":
    unittest.main(verbosity=2)
