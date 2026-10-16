"""Versioned experiment settings and archived tuning provenance."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG = ROOT / "configs/paper.json"
TARGET_NAMES = ("Teff", "[Fe/H]", "log g")
OUTPUT_NAMES = ("teff_output", "feh_output", "logg_output")
SPLITS = (("train", "train"), ("validation", "val"), ("test", "test"))


def load_config(path=None, **overrides):
    config = json.loads(Path(path or PAPER_CONFIG).read_text(encoding="utf-8"))
    config = deepcopy(config)
    config.update({k: v for k, v in overrides.items() if v is not None})
    for key in ("epochs", "batch_size", "threads", "input_pixels"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if config["input_pixels"] != 4000:
        raise ValueError("This benchmark requires 4000 wavelength pixels.")
    if config["reference_labels"] != "inverse_scaled_float64_v1":
        raise ValueError("Unsupported reference-label convention.")
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer.")
    return config


def load_tuning_history(source=None):
    source = Path(source or ROOT / "reference/tuning_history.json")
    if source.is_dir():
        records = []
        for p in sorted(source.glob("trial_*/trial.json")):
            record = json.loads(p.read_text(encoding="utf-8"))
            record["trial_id"] = p.parent.name
            records.append(record)
    else:
        records = json.loads(source.read_text(encoding="utf-8"))["trials"]
    if not records:
        raise ValueError(f"No archived tuning records: {source}")
    return records
