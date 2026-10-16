"""Reports derive from one aligned prediction set, with comparisons explicitly requested."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TARGET_NAMES
from .predictions import assert_aligned, physical_labels, read_predictions
from .runs import check_derived_output, derived_output, sha256, write_json


def evaluate(run, data, *, comparators=(), output_dir=None, replace=False, n_boot=10000):
    if not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("Bootstrap count must be a positive integer.")
    output = check_derived_output(output_dir or run.path / "reports", replace=replace)
    base = read_predictions(run.path / "test_predictions.npz")
    np.testing.assert_array_equal(base["test_ids"], data["test_ids"])
    np.testing.assert_array_equal(base["y_true"], physical_labels(data))
    others = {}
    for name in comparators:
        path = run.path / "baselines" / name / "test_predictions.npz"
        other = read_predictions(path)
        assert_aligned(base, other)
        others[name] = other
    model = run.path / "retrained_paper_model.keras"
    config = json.loads((run.path / "run_config.json").read_text(encoding="utf-8"))
    if config.get("model_sha256") and sha256(model) != config["model_sha256"]:
        raise ValueError("Checkpoint checksum differs from the trained run configuration.")
    from .training import runtime
    runtime(config)
    from results_and_evaluations.bootstrap_paper_model_mae import bootstrap_mae
    from results_and_evaluations.paired_bootstrap_all_models import bootstrap_paired_differences
    from results_and_evaluations.measure_madds import estimate_model_flops
    from scripts.analyze_parameter_ranges import summarize
    errors = abs(base["y_pred"] - base["y_true"])
    result = {"n_test": len(errors), "n_boot": n_boot, "seed": 42,
              "model_sha256": sha256(model), "predictions_sha256": sha256(run.path / "test_predictions.npz"),
              "dataset_id": str(data["dataset_id"].item()), "reference_labels": "inverse_scaled_float64_v1",
              "metrics": {}, "comparisons": {}, "comparison_status": "requested" if others else "not_requested"}
    for i, target in enumerate(TARGET_NAMES):
        mean, low, high = bootstrap_mae(errors[:, i], n_boot, 42)
        result["metrics"][target] = {"mae": float(mean), "ci_95": [float(low), float(high)]}
    if others:
        differences = np.stack([abs(v["y_pred"] - base["y_true"]) - errors for v in others.values()], axis=2)
        means, intervals = bootstrap_paired_differences(differences, n_boot=n_boot, seed=42)
        for k, name in enumerate(others):
            result["comparisons"][name] = {target: {
                "mae_difference": float(means[i, k]),
                "ci_95": [float(intervals[0, i, k]), float(intervals[1, i, k])],
            } for i, target in enumerate(TARGET_NAMES)}
    params, flops, macs = estimate_model_flops(model, 4000)
    architecture = {"model_sha256": result["model_sha256"], "trainable_parameters": params,
                    "flops": flops, "actual_heads": config["heads"]}
    ranges = summarize(base["y_true"], base["y_pred"], n_boot=n_boot, seed=42)
    output = Path(output_dir or run.path / "reports")
    with derived_output(output, replace=replace) as temp:
        write_json(temp / "evaluation.json", result)
        write_json(temp / "bootstrap_mae.json", result)
        write_json(temp / "paired_bootstrap.json", {"n_test": len(errors), "comparisons": result["comparisons"], "status": result["comparison_status"]})
        write_json(temp / "architecture_audit.json", architecture)
        ranges.to_csv(temp / "parameter_range_errors.csv", index=False)
        write_json(temp / "parameter_range_errors.json", {"n_test": len(errors), "n_boot": n_boot,
                   "seed": 42, "rows": json.loads(ranges.to_json(orient="records", double_precision=15))})
        pd.DataFrame([{"model": "proposed", "params": params, "flops": flops, "macs": macs}]).to_csv(temp / "complexity.csv", index=False)
        rows = [{"model": "proposed", "target": target, **result["metrics"][target]} for target in TARGET_NAMES]
        for name, other in others.items():
            rows += [{"model": name, "target": target, "mae": float(np.mean(abs(other["y_pred"][:, i] - base["y_true"][:, i])))} for i, target in enumerate(TARGET_NAMES)]
        pd.DataFrame(rows).to_csv(temp / "test_metrics.csv", index=False)
        write_json(temp / "experiment_summary.json", {
            "configuration": config, "dataset_id": result["dataset_id"],
            "counts": {s: int(len(data[f"{s}_ids"])) for s in ("train", "val", "test")},
            "complexity": architecture,
        })
    print(f"Evaluation complete: {output}", flush=True)
    return result


def ablation_reports(run, *, n_boot=10000, output_dir=None, replace=False):
    out = check_derived_output(output_dir or run.path / "reports/ablations", replace=replace)
    from results_and_evaluations.paired_bootstrap_all_models import bootstrap_paired_differences
    base = read_predictions(run.path / "test_predictions.npz")
    choices = {}
    params = {}
    for name in ("no_noise_augmentation", "no_skip", "no_layernorm", "no_residual_no_layernorm"):
        folder = run.path / "ablations" / name
        item = read_predictions(folder / "test_predictions.npz")
        assert_aligned(base, item)
        choices[name] = item["y_pred"]
        detail = json.loads((folder / "metrics.json").read_text())
        params[name] = detail.get("params", detail.get("trainable_parameters"))
    singles, count = [], 0
    for i, name in enumerate(("single_teff", "single_feh", "single_logg")):
        folder = run.path / "ablations" / name
        item = read_predictions(folder / "test_predictions.npz")
        np.testing.assert_array_equal(item["test_ids"], base["test_ids"])
        np.testing.assert_array_equal(item["y_true"], base["y_true"][:, i:i+1])
        singles.append(item["y_pred"][:, 0])
        info = json.loads((folder / "metrics.json").read_text())
        count += info.get("params", info.get("trainable_parameters"))
    choices["no_multitask"] = np.column_stack(singles)
    params["no_multitask"] = count
    errors = abs(base["y_pred"] - base["y_true"])
    differences = np.stack([abs(pred-base["y_true"])-errors for pred in choices.values()], axis=2)
    mean, ci = bootstrap_paired_differences(differences, n_boot=n_boot, seed=42)
    rows = [{"variant": name, "params":params[name], "target":target,
             "mae":float(np.mean(abs(pred[:,i]-base["y_true"][:,i]))),
             "mae_difference":float(mean[i,k]), "ci_low":float(ci[0,i,k]), "ci_high":float(ci[1,i,k])}
            for k,(name,pred) in enumerate(choices.items()) for i,target in enumerate(TARGET_NAMES)]
    with derived_output(out, replace=replace) as temporary:
        pd.DataFrame(rows).to_csv(temporary / "ablation_summary.csv", index=False)
        write_json(temporary / "ablation_paired_bootstrap.json", {"n_test":len(errors),"n_boot":n_boot,
            "seed":42,"rows":rows,"difference":"MAE_variant - MAE_proposed"})
    return rows
