"""One physical-label convention and aligned, portable prediction archives."""
from pathlib import Path
import numpy as np

from .config import TARGET_NAMES


def physical_labels(data, split="test"):
    short = {"validation": "val", "train": "train", "val": "val", "test": "test"}[split]
    return (np.asarray(data[f"y_{short}_targets"], dtype=np.float64)
            * np.asarray(data["label_robust_scale"], dtype=np.float64)
            + np.asarray(data["label_robust_center"], dtype=np.float64))


def physical_predictions(predictions, data):
    from results_and_evaluations.prediction_utils import prediction_matrix
    values = prediction_matrix(predictions).astype(np.float64)
    return (values * np.asarray(data["label_robust_scale"], dtype=np.float64)
            + np.asarray(data["label_robust_center"], dtype=np.float64))


def _validated(values):
    ids = np.asarray(values["test_ids"])
    names = tuple(np.asarray(values["target_names"]).tolist())
    true, pred = (np.asarray(values[k], dtype=np.float64) for k in ("y_true", "y_pred"))
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("Prediction IDs must be unique integer identifiers.")
    if not names or tuple(t for t in TARGET_NAMES if t in names) != names:
        raise ValueError("Invalid target names/order.")
    if true.shape != (len(ids), len(names)) or pred.shape != true.shape:
        raise ValueError("Prediction/reference shapes do not match IDs and target names.")
    if not len(ids) or not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Prediction/reference arrays must be nonempty and finite.")
    fingerprint = str(np.asarray(values.get("dataset_id", "")).item())
    return dict(test_ids=ids.astype(np.int64), target_names=np.asarray(names),
                y_true=true, y_pred=pred, dataset_id=fingerprint)


def write_predictions(path, ids, true, pred, *, dataset_id, target_names=TARGET_NAMES):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Preserve existing predictions: {path}")
    values = _validated(dict(test_ids=ids, target_names=target_names, y_true=true,
                             y_pred=pred, dataset_id=dataset_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)
    return path


def read_predictions(path):
    with np.load(path, allow_pickle=False) as data:
        return _validated(dict(data))


def assert_aligned(reference, other):
    if not np.array_equal(reference["test_ids"], other["test_ids"]):
        raise ValueError("Prediction IDs or object order differ.")
    if not np.array_equal(reference["target_names"], other["target_names"]):
        raise ValueError("Prediction target order differs.")
    if not np.array_equal(reference["y_true"], other["y_true"]):
        raise ValueError("Physical reference labels differ.")
    if reference["dataset_id"] and other["dataset_id"] and reference["dataset_id"] != other["dataset_id"]:
        raise ValueError("Prediction dataset fingerprints differ.")
