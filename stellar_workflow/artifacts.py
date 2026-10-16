"""Checksummed reference releases, without historical absolute-path dependencies."""
import json
from pathlib import Path
import shutil

from .config import ROOT, load_config
from .datasets import download_verified, prepare_public, load_compact
from .runs import Run, sha256, write_json


def _member(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if path == root or not path.is_relative_to(root) or Path(name).is_absolute():
        raise ValueError(f"Artifact path is outside the release: {name}")
    return path


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported artifact manifest.")
    for name, expected in manifest["files"].items():
        path = _member(root, name)
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Artifact missing or size changed: {name}")
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"Artifact checksum changed: {name}")
    return manifest


def install_bundle(source, destination):
    manifest = verify_bundle(source)
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Reference destination must be empty; archived experiments are preserved.")
    run = Run(destination)
    with run.stage("reference-import", list(manifest["files"]) + ["artifact_manifest.json"]):
        for name in [*manifest["files"], "artifact_manifest.json"]:
            target = _member(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_member(source, name), target)
    verify_bundle(destination)
    return run


def fetch_reference(*, offline=False):
    pointer = json.loads((ROOT / "reference/artifacts.json").read_text(encoding="utf-8"))
    if not pointer.get("revision") or not pointer.get("manifest_sha256"):
        raise RuntimeError("The model release has not been published/pinned yet. Use reproduce-paper --bundle PATH for the staged release.")
    root = ROOT / "outputs/reference-cache" / pointer["revision"]
    prefix = f"https://huggingface.co/{pointer['repo_id']}/resolve/{pointer['revision']}"
    download_verified(prefix + "/artifact_manifest.json", root / "artifact_manifest.json", pointer["manifest_sha256"], offline=offline)
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    for name, detail in manifest["files"].items():
        download_verified(prefix + "/" + name, _member(root, name), detail["sha256"], offline=offline)
    verify_bundle(root)
    return root


def reproduce_paper(args):
    from .pipeline import figures, initialize
    from .evaluation import evaluate, ablation_reports
    from .audit import audit
    source = args.bundle or fetch_reference(offline=args.offline)
    manifest = verify_bundle(source)
    if manifest.get("release") != "paper-v1":
        raise ValueError("Expected the immutable paper-v1 reference release.")
    run = install_bundle(source, args.run_dir)
    config = load_config(run.path / "run_config.json")
    path = prepare_public(args.data_dir, offline=args.offline)
    initialize(run, config, path)
    with load_compact(path) as data:
        evaluate(run, data, comparators=("cnn", "dnn", "transformer"), n_boot=args.n_boot)
        ablation_reports(run, n_boot=args.n_boot)
        figures(run, path, paper_reference=True, paper_dir=args.paper_dir, n_boot=args.n_boot)
        audit(run, data)
    print(f"Archived paper reproduced without retraining: {run.path}", flush=True)
