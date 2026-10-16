"""Portable numerical audit; optional Torch inference is explicitly distinguished."""
import ast
import json
from pathlib import Path

import numpy as np

from .config import ROOT, TARGET_NAMES
from .predictions import read_predictions, assert_aligned, physical_labels, physical_predictions
from .runs import check_derived_output, derived_output, sha256, write_json


def audit(run, data, *, verify_transformer=False, output_dir=None, replace=False, comparators=()):
    check_derived_output(output_dir or run.path / "audit", replace=replace)
    from .training import runtime
    config = json.loads((run.path / "run_config.json").read_text())
    tf = runtime(config)
    for name in comparators:
        if not (run.path / "baselines" / name / "test_predictions.npz").exists():
            raise FileNotFoundError(f"Requested audit comparator is missing: {name}")
    for name, detail in run.read()["artifacts"].items():
        if sha256(run.resolve(name)) != detail["sha256"]:
            raise ValueError(f"Run artifact checksum changed: {name}")
    base = read_predictions(run.path / "test_predictions.npz")
    np.testing.assert_array_equal(base["test_ids"], data["test_ids"])
    np.testing.assert_array_equal(base["y_true"], physical_labels(data))
    report = {"dataset_id": str(data["dataset_id"].item()), "n_test": len(base["test_ids"]), "models": {}}
    sample = data["X_test_features"]
    model = tf.keras.models.load_model(run.path / "retrained_paper_model.keras", compile=False)
    prediction = physical_predictions(model.predict(sample, batch_size=256, verbose=0), data)
    np.testing.assert_allclose(prediction, base["y_pred"][:len(sample)], atol=.002, rtol=1e-6)
    report["models"]["proposed"] = {"status": "inference_verified", "objects":len(sample),
        "params":int(model.count_params()), "max_abs_difference":np.max(abs(prediction-base["y_pred"][:len(sample)]),axis=0).tolist()}
    report["models"]["proposed"]["inference_mae"] = dict(zip(TARGET_NAMES, np.mean(abs(prediction-base["y_true"]),axis=0).tolist()))
    del model
    for name in ("ols", "ridge", "cnn", "dnn", "transformer"):
        folder = run.path / "baselines" / name
        if not (folder / "test_predictions.npz").exists():
            continue
        other = read_predictions(folder / "test_predictions.npz")
        assert_aligned(base, other)
        status = {"status": "predictions_verified"}
        if name == "transformer":
            status["inference"] = "not_run_optional_dependency"
            if verify_transformer:
                from finetuning.checkpoint_io import load_checkpoint
                from finetuning.train_spectral_transformer import SpectralTransformerRegressor
                import torch
                torch.set_num_threads(config.get("threads", 2))
                checkpoint = load_checkpoint(folder / "best_model.pt", map_location="cpu")
                args = checkpoint["args"]
                transformer = SpectralTransformerRegressor(
                    input_dim=4000, patch_size=args["patch_size"], d_model=args["d_model"],
                    num_layers=args["layers"], num_heads=args["heads"], mlp_ratio=args["mlp_ratio"], dropout=args["dropout"])
                transformer.load_state_dict(checkpoint["model"])
                transformer.eval()
                # Archived Transformer inference is checked on a deterministic subset:
                # full saved-prediction metrics above use every test object.
                transformer_sample = sample[:64]
                with torch.no_grad():
                    scaled = transformer(torch.from_numpy(transformer_sample)).float().numpy()
                archived_scaled = (other["y_pred"][:len(transformer_sample)] - data["label_robust_center"]) / data["label_robust_scale"]
                np.testing.assert_allclose(scaled, archived_scaled, atol=.02, rtol=.005)
                status.update(inference="verified_cpu_float32", tolerance_scaled={"atol":.02,"rtol":.005},
                              archived_precision="CUDA bfloat16 when enabled", selected_epoch=checkpoint["epoch"],
                              inference_objects=len(transformer_sample),
                              max_scaled_difference=np.max(abs(scaled-archived_scaled),axis=0).tolist())
                del transformer
        elif name in ("cnn", "dnn"):
            if name == "cnn":
                model = tf.keras.models.load_model(folder / "fabbro_cnn.keras", compile=False)
                raw = model.predict(sample[..., None], batch_size=256, verbose=0)
                del model
            else:
                columns = []
                for target in ("teff", "feh", "logg"):
                    model = tf.keras.models.load_model(folder / f"li_dnn_{target}.keras", compile=False)
                    columns.append(model.predict(sample, batch_size=256, verbose=0).ravel())
                    del model
                raw = np.column_stack(columns)
            forward = physical_predictions(raw, data)
            np.testing.assert_allclose(forward, other["y_pred"][:len(sample)], atol=.002, rtol=1e-6)
            status["status"] = "inference_verified"
            status["objects"] = len(sample)
            status["max_abs_difference"] = np.max(abs(forward-other["y_pred"]),axis=0).tolist()
            status["inference_mae"] = dict(zip(TARGET_NAMES, np.mean(abs(forward-other["y_true"]),axis=0).tolist()))
        report["models"][name] = status
    if verify_transformer and "transformer" not in report["models"]:
        raise FileNotFoundError("Transformer verification requested but its artifacts are absent.")
    for p in sorted((run.path / "ablations").glob("*/test_predictions.npz")):
        other = read_predictions(p)
        np.testing.assert_array_equal(other["test_ids"], base["test_ids"])
        indices = [TARGET_NAMES.index(t) for t in other["target_names"]]
        np.testing.assert_array_equal(other["y_true"], base["y_true"][:, indices])
    sources = [p for folder in ("stellar_workflow","training","model_definitions","data_acquisition","data_preprocessing","results_and_evaluations","scripts","finetuning") for p in (ROOT/folder).glob("*.py")]
    for p in sources:
        ast.parse(p.read_text(encoding="utf-8-sig"))
    report["python_sources_parsed"] = len(sources)
    report["test_mae"] = dict(zip(TARGET_NAMES,np.mean(abs(base["y_pred"]-base["y_true"]),axis=0).tolist()))
    with derived_output(Path(output_dir or run.path / "audit"), replace=replace) as out:
        write_json(out / "research_audit.json", report)
    print("Artifact, label, feature and inference audit passed.", flush=True)
    return report
