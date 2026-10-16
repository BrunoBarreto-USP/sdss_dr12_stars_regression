"""Thin orchestration of data, training, reporting and reference reproduction."""
import json
import os
from pathlib import Path
import tempfile

from .config import ROOT, load_config
from .datasets import COMPACT_NAME, build_compact, load_compact, prepare_public
from .runs import Run, derived_output, write_json


def run_data(args, run=None):
    reference = run.path / "data_reference.json" if run else None
    if reference is not None and reference.exists():
        info = json.loads(reference.read_text(encoding="utf-8"))
        candidate = (run.path / info["cache_relative"] / COMPACT_NAME).resolve()
        if candidate.exists():
            return candidate
        if info["source_kind"] == "fixture":
            raise FileNotFoundError("This smoke run's synthetic data cache is missing.")
        return prepare_public(args.data_dir, row_limit=info["row_limit"], offline=args.offline)
    return prepare_public(args.data_dir, row_limit=args.row_limit, offline=args.offline)


def run_config(args, run=None):
    if args.config:
        source = args.config
    elif run and (run.path / "run_config.json").exists():
        source = run.path / "run_config.json"
    elif run and (run.path / "experiment_config.json").exists():
        source = run.path / "experiment_config.json"
    else:
        source = None
    return load_config(source, **{k: getattr(args, k, None) for k in ("epochs", "batch_size", "threads", "seed")})


def initialize(run, config, path):
    from filelock import FileLock
    from .datasets import _manifest_path
    with FileLock(run.path / ".stage.lock", timeout=0):
        manifest = json.loads(_manifest_path(path).read_text(encoding="utf-8"))
        info = {"dataset_id": manifest["dataset_id"], "source_kind": manifest["source_kind"],
                "row_limit": manifest["row_limit"], "cache_relative": Path(os.path.relpath(path.parent, run.path)).as_posix()}
        old = run.path / "data_reference.json"
        if old.exists() and json.loads(old.read_text(encoding="utf-8"))["dataset_id"] != info["dataset_id"]:
            raise ValueError("This run already belongs to a different dataset fingerprint.")
        cfg_path = run.path / "experiment_config.json"
        if cfg_path.exists():
            previous = json.loads(cfg_path.read_text(encoding="utf-8"))
            for key in ("seed", "epochs", "batch_size", "augmentation"):
                if config[key] != previous[key]:
                    raise ValueError(f"Run protocol differs for {key}; choose a fresh run.")
        else:
            write_json(cfg_path, config)
        write_json(old, info)


def make_fixture(root, n=8):
    """Clearly labeled synthetic spectra for an offline execution smoke, never paper results."""
    import numpy as np
    import pandas as pd
    from .datasets import wavelength_grid
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    grid = np.log10(wavelength_grid())
    rng = np.random.default_rng(42)
    for split_index, split in enumerate(("train", "validation", "test")):
        path = root / f"{split}.parquet"
        if path.exists():
            continue
        raw = 20 + rng.normal(size=(n, 4000)) + np.sin(np.linspace(0, 20, 4000))[None, :]
        processed = ((raw - raw.mean(axis=1, keepdims=True)) / raw.std(axis=1, keepdims=True)).astype(np.float32)
        pd.DataFrame({
            "spec_id": np.arange(split_index * n, (split_index + 1) * n, dtype=np.int64),
            "plate": 1, "mjd": 1, "fiber": np.arange(n), "ra": rng.uniform(0, 360, n),
            "dec": rng.uniform(-60, 60, n), "snr": np.linspace(12, 120, n), "rv_adop": np.linspace(-100, 100, n),
            "catalog_teff": np.linspace(4400, 7200, n), "catalog_feh": np.linspace(-2, .2, n),
            "catalog_logg": np.linspace(2, 4.5, n), "processed_flux": list(processed),
            "flux": list(raw), "loglam": [grid] * n, "ivar": [np.ones(4000)] * n,
            "mask": [np.zeros(4000, dtype=bool)] * n,
        }).to_parquet(path, index=False)
    return build_compact(root, source_kind="fixture")


def figures(run, path, *, output_dir=None, replace=False, paper_reference=False, paper_dir=None, n_boot=10000, only_training=False):
    required = ("retrained_paper_model.keras", "test_predictions.npz", "history.csv", "run_config.json")
    for name in required:
        if not (run.path / name).exists():
            raise FileNotFoundError(f"Figure input missing: {name}")
    from scripts.generate_paper_figures import Figures
    from scripts.analyze_parameter_ranges import plot, summarize
    from .predictions import read_predictions
    target = Path(output_dir or run.path / ("training-figures" if only_training else "figures"))
    with derived_output(target, replace=replace) as temporary:
        import pyarrow.parquet as pq
        count = pq.ParquetFile(path.parent / "train.parquet").metadata.num_rows
        f = Figures(ROOT, temporary, paper_dir, run.path, data_dir=path.parent,
                    strict_paper=paper_reference, dpi=120 if count < 64 else 600)
        if only_training:
            f.training()
            f.finish()
            return
        f.load_metadata()
        f.sky()
        f.observations()
        f.distributions()
        f.raw_diagnostics()
        f.optimization()
        f.weights()
        f.predictions()
        f.residual_diagram()
        f.model_diagram()
        f.training()
        pred = read_predictions(run.path / "test_predictions.npz")
        frame = summarize(pred["y_true"], pred["y_pred"], n_boot=n_boot, seed=42)
        plot(f, frame, n_boot=n_boot, seed=42)
        f.finish()
    print(f"Figures complete: {target}", flush=True)


def smoke(args):
    from .training import train_primary
    from .evaluation import evaluate
    if args.run_dir is None:
        (ROOT / "outputs").mkdir(exist_ok=True)
        args.run_dir = Path(tempfile.mkdtemp(prefix="smoke-", dir=ROOT / "outputs"))
    run = Run(args.run_dir)
    config = load_config(epochs=1, batch_size=8, threads=2, seed=42)
    config["profile"] = "smoke"
    if args.online:
        path = prepare_public(run.path / "data", row_limit=8, offline=False)
    else:
        path = make_fixture(run.path / "data")
    initialize(run, config, path)
    with load_compact(path) as data:
        train_primary(run, config, data)
        evaluate(run, data, n_boot=min(args.n_boot, 64))
        figures(run, path, n_boot=min(args.n_boot, 64))
    if args.extended:
        extended_smoke(run, config, path)
    print(f"Smoke execution passed ({'public sample' if args.online else 'synthetic fixture'}): {run.path}", flush=True)


def extended_smoke(run, config, path):
    from .training import train_ablations, train_baselines, tune_primary
    with load_compact(path) as data:
        train_baselines(run, config, data, models=("ols", "ridge", "cnn", "dnn"), pretrain_epochs=1)
        train_ablations(run, config, data)
        other = Run(run.path / "tuning_smoke")
        initialize(other, config, path)
        tune_primary(other, config, data, max_trials=1, search_epochs=1)


def dispatch(args):
    if args.command == "smoke":
        return smoke(args)
    if args.command == "prepare-data":
        print(prepare_public(args.data_dir, row_limit=args.row_limit, offline=args.offline))
        return
    if args.command == "reproduce-paper":
        from .artifacts import reproduce_paper
        return reproduce_paper(args)
    run = Run(args.run_dir)
    config = run_config(args, run)
    path = run_data(args, run)
    initialize(run, config, path)
    with load_compact(path) as data:
        if args.command == "train":
            from .training import train_primary
            train_primary(run, config, data)
            from .evaluation import evaluate
            evaluate(run, data)
        elif args.command == "tune":
            from .training import tune_primary
            tune_primary(run, config, data, max_trials=args.max_trials, search_epochs=args.search_epochs)
        elif args.command == "baselines":
            from .training import train_baselines
            train_baselines(run, config, data, models=args.models, allow_cpu=args.allow_cpu, pretrain_epochs=args.pretrain_epochs)
        elif args.command == "ablations":
            from .training import train_ablations
            train_ablations(run, config, data, variants=args.variants, low_memory=args.low_memory)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            evaluate(run, data, comparators=args.comparators, output_dir=args.output_dir, replace=args.replace_derived, n_boot=args.n_boot)
        elif args.command == "figures":
            figures(run, path, output_dir=args.output_dir, replace=args.replace_derived,
                    paper_reference=args.paper_reference, paper_dir=args.paper_dir,
                    n_boot=args.n_boot, only_training=args.only_training)
        elif args.command == "audit":
            from .audit import audit
            audit(run, data, verify_transformer=args.verify_transformer,
                  output_dir=args.output_dir, replace=args.replace_derived, comparators=args.comparators)
