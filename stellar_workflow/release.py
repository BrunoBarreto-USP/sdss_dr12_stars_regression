"""Maintainer-only staging of existing research artifacts. Never trains or uploads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .artifacts import verify_bundle
from .config import ROOT, load_config, TARGET_NAMES
from .datasets import REFERENCE
from .runs import sha256, write_json


def stage_release(source_root, output_dir):
    source, output = Path(source_root).resolve(), Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Release staging requires an empty directory.")
    if output == source or source.is_relative_to(output):
        raise ValueError("Release output cannot contain the source experiment.")
    experiment = source / "outputs/paper_figures/single_column/corrected_best_trial"
    files = {}
    for path in experiment.rglob("*"):
        if path.is_file() and path.suffix in {".keras", ".h5", ".npz", ".json", ".csv"}:
            files[path.relative_to(experiment).as_posix()] = path
    for path in (source / "models/benchmarks").glob("*.keras"):
        model = "cnn" if path.name.startswith("fabbro") else "dnn"
        files[f"baselines/{model}/{path.name}"] = path
    for model in ("cnn", "dnn"):
        files[f"baselines/{model}/test_predictions.npz"] = source / f"data/{model}_test_predictions.npz"
    for path in (source / "results_and_evaluations/spectral_transformer").glob("*"):
        if path.is_file() and path.suffix in {".pt", ".npz", ".json", ".csv"}:
            files[f"baselines/transformer/{path.name}"] = path
    for path in (source / "results_and_evaluations/keras_tuner/stellar_baseline_orchestration").glob("trial_*/trial.json"):
        files[f"original_tuner/{path.parent.name}/trial.json"] = path
    diagnostic = source / "outputs/paper_figures/single_column/raw_flux_diagnostics.npz"
    if diagnostic.exists():
        files["diagnostics/raw_flux_diagnostics.npz"] = diagnostic
    if not files or not all(p.is_file() for p in files.values()):
        raise FileNotFoundError("The complete original experiment is required for release staging.")
    if len([n for n in files if n.startswith("original_tuner/")]) != 100:
        raise ValueError("Expected all 100 original tuner records.")
    output.mkdir(parents=True, exist_ok=True)
    inverse = {str(p).replace("\\", "/"): name for name, p in files.items()}

    def portable(value):
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str):
            normalized = value.replace("\\", "/")
            if normalized in inverse:
                return inverse[normalized]
            for prefix in (source.as_posix() + "/", "/workspace/spect/"):
                if normalized.startswith(prefix):
                    return normalized[len(prefix):]
        return value

    original_hashes = {}
    for name, path in sorted(files.items()):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        original_hashes[name] = sha256(path)
        if path.suffix == ".json" and not name.startswith("original_tuner/"):
            write_json(target, portable(json.loads(path.read_text(encoding="utf-8"))))
        elif path.suffix == ".csv":
            content = path.read_text(encoding="utf-8")
            for original_path, new_path in inverse.items():
                content = content.replace(original_path, new_path).replace(original_path.replace("/", "\\"), new_path)
            target.write_text(content, encoding="utf-8", newline="\n")
        else:
            shutil.copy2(path, target)
            if sha256(target) != original_hashes[name]:
                raise ValueError(f"Binary artifact changed during copy: {name}")
    identity = dict(schema_version=1, source_kind="public", repo_id=REFERENCE["repo_id"],
                    revision=REFERENCE["revision"], split_files=REFERENCE["splits"], row_limit=None,
                    grid=REFERENCE["grid"], normalize_flux=False, reference_labels="inverse_scaled_float64_v1")
    dataset_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    config = load_config()
    original = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    augmentation = {**config["augmentation"], **original.get("augmentation", {})}
    config.update(original, profile="paper_reference", augmentation=augmentation, dataset_id=dataset_id,
                  model_sha256=sha256(output / "retrained_paper_model.keras"))
    write_json(output / "run_config.json", config)
    import numpy as np
    with np.load(source / "data/sdss_dr12_processed_flux_benchmark.npz") as data:
        scaler = {"center": data["label_robust_center"].tolist(), "scale": data["label_robust_scale"].tolist()}
    write_json(output / "target_scaler.json", {**scaler, "target_names": TARGET_NAMES,
               "grid": REFERENCE["grid"], "input": "published processed_flux, float32, per-spectrum standardization"})
    write_json(output / "dataset_reference.json", REFERENCE)
    shutil.copy2(ROOT / "reference/tuning_history.json", output / "tuning_history.json")
    shutil.copy2(ROOT / "LICENSE", output / "LICENSE")
    (output / "README.md").write_text(MODEL_CARD, encoding="utf-8", newline="\n")
    manifest = {"schema_version": 1, "release": "paper-v1", "dataset": REFERENCE,
                "dataset_id": dataset_id, "training_source_commit": None,
                "known_issues": ["Archived Transformer selection used unweighted batch means; preserved for provenance.",
                                 "Published processed_flux retains the original native-pixel mask policy."], "files": {}}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            name = path.relative_to(output).as_posix()
            detail = {"bytes": path.stat().st_size, "sha256": sha256(path)}
            if name in original_hashes:
                detail["source_sha256"] = original_hashes[name]
            manifest["files"][name] = detail
    write_json(output / "artifact_manifest.json", manifest)
    verify_bundle(output)
    write_json(ROOT / "reference/artifacts.json", {"repo_id": REFERENCE["repo_id"], "repo_type": "model",
               "release": "paper-v1", "revision": None, "manifest_sha256": sha256(output / "artifact_manifest.json"),
               "publication_status": "staged_not_published"})
    print(f"Verified {len(manifest['files'])} files, {sum(v['bytes'] for v in manifest['files'].values()):,} bytes: {output}")
    return manifest


MODEL_CARD = """---
license: mit
library_name: keras
pipeline_tag: tabular-regression
datasets:
- BrunoBarreto/sdss_dr12_stars_regression
tags:
- astronomy
- stellar-spectra
---

# SDSS DR12 stellar regression: paper-v1

Frozen reference experiment for Stellar Parameter Estimation Using Residual Multitask Neural Networks.
The main residual multitask model has 530,275 parameters, uses trial 099 and seed 42,
and selects epoch 74 from an 80-epoch training budget. This release preserves the
original checkpoint and prediction arrays. Source software fixes do not imply a new experiment.

## Inputs and outputs

Input: one float32 vector of 4,000 published processed_flux values on the logarithmic
wavelength grid log10(lambda/Angstrom) = linspace(3.5754, 3.9670, 4000), with the
published rest-frame resampling and per-spectrum standardization. Arbitrary raw flux
cannot be passed directly. Output heads: teff_output, feh_output, logg_output, in scaled
target space. Convert each column to physical units using target_scaler.json: value * scale + center.
Targets are Teff (K), [Fe/H] (dex), log g (dex), in that order.

## Reproduction

Install the accompanying source repository and its pinned Python 3.11 requirements.
Run python -m stellar_workflow reproduce-paper --run-dir outputs/reference/paper-v1.
The source pins this model repository to an immutable revision after publication.
An already downloaded bundle can instead be supplied with --bundle PATH.
Use python -m stellar_workflow audit --run-dir PATH --verify-transformer with the optional Torch dependency
to additionally infer from the archived Transformer checkpoint.

## Recorded results and scope

Main-model test MAE on 14,443 spectra: 58.717678 K, 0.101314 dex, and 0.131862 dex.
The bundle also contains CNN, DNN and Transformer checkpoints/predictions, seven
ablations, histories, bootstrap/complexity/parameter-bin reports, and all 100 original tuner records.
artifact_manifest.json lists portable-copy hashes and original source hashes.
The original training source commit was not recorded and is explicitly unknown.

## Limitations and licenses

Evaluation uses one archived seed and SSPP catalogue labels, not independent physical ground truth.
The archived Transformer used unweighted validation batch means for checkpoint selection;
that known limitation is retained here. New source runs use object-weighted validation loss
and have separate provenance. The original dataset's native-pixel masking policy is preserved.
Reproduction begins with the published benchmark, not reconstruction of the original survey download.

Source and authored model artifacts are MIT licensed (LICENSE). The separately hosted dataset
is CC-BY-4.0: cite its dataset card and the paper, retaining the original SDSS/SSPP attribution.
The dataset license is not replaced by the model license.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    stage_release(args.source_root, args.output_dir)


if __name__ == "__main__":
    main()
