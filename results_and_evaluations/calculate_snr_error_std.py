"""Calculate prediction-error statistics in fixed SNR bins."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "stellar_regression_model.keras"
DEFAULT_DATA = PROJECT_ROOT / "data" / "sdss_dr12_processed_flux_benchmark.npz"
DEFAULT_SNR_DATA = PROJECT_ROOT / "data" / "test.parquet"
DEFAULT_OUT = (
    PROJECT_ROOT
    / "results_and_evaluations"
    / "stellar_regression_model_snr_error_stats.csv"
)
TARGETS = ("Teff", "FeH", "logg")


def stack_predictions(predictions: object) -> np.ndarray:
    if isinstance(predictions, (list, tuple)):
        return np.column_stack(
            [np.asarray(prediction).reshape(-1) for prediction in predictions]
        )
    return np.asarray(predictions)


def load_prediction_scaler(
    data: np.lib.npyio.NpzFile,
    scaler_path: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    if scaler_path is None:
        return (
            np.asarray(data["label_robust_center"], dtype=np.float64),
            np.asarray(data["label_robust_scale"], dtype=np.float64),
        )

    with np.load(scaler_path, allow_pickle=True) as scaler:
        if {"centers", "scales"}.issubset(scaler.files):
            return (
                np.asarray(scaler["centers"], dtype=np.float64),
                np.asarray(scaler["scales"], dtype=np.float64),
            )
        if {"label_robust_center", "label_robust_scale"}.issubset(scaler.files):
            return (
                np.asarray(scaler["label_robust_center"], dtype=np.float64),
                np.asarray(scaler["label_robust_scale"], dtype=np.float64),
            )
    raise KeyError(f"Could not find target centers and scales in {scaler_path}")


def calculate_stats(args: argparse.Namespace) -> pd.DataFrame:
    with np.load(args.data, allow_pickle=True) as data:
        x_test = np.asarray(data["X_test_features"], dtype=np.float32)
        test_ids = np.asarray(data["test_ids"], dtype=np.int64)
        centers, scales = load_prediction_scaler(data, args.prediction_scalers)

        if "y_test_targets_original" in data.files:
            y_true = np.asarray(data["y_test_targets_original"], dtype=np.float64)
        else:
            y_test_scaled = np.asarray(data["y_test_targets"], dtype=np.float64)
            reference_centers = np.asarray(data["label_robust_center"], dtype=np.float64)
            reference_scales = np.asarray(data["label_robust_scale"], dtype=np.float64)
            y_true = y_test_scaled * reference_scales + reference_centers

    model = tf.keras.models.load_model(
        args.model,
        compile=False,
        safe_mode=False,
    )
    predictions_scaled = stack_predictions(
        model.predict(x_test, batch_size=args.batch_size, verbose=1)
    )
    if predictions_scaled.shape != y_true.shape:
        raise ValueError(
            f"Prediction shape {predictions_scaled.shape} does not match "
            f"reference shape {y_true.shape}"
        )
    y_pred = predictions_scaled.astype(np.float64) * scales + centers

    snr_frame = pd.read_parquet(args.snr_data, columns=["spec_id", "snr"])
    if snr_frame["spec_id"].duplicated().any():
        raise ValueError(f"Duplicate spec_id values found in {args.snr_data}")
    snr_lookup = snr_frame.set_index("spec_id")["snr"]
    missing_ids = np.setdiff1d(test_ids, snr_lookup.index.to_numpy())
    if missing_ids.size:
        raise ValueError(
            f"{missing_ids.size} test IDs have no SNR value in {args.snr_data}"
        )
    snr = snr_lookup.loc[test_ids].to_numpy(dtype=np.float64)

    residuals = y_pred - y_true
    absolute_errors = np.abs(residuals)
    rows: list[dict[str, float | int | str]] = []

    for lower in range(args.snr_min, args.snr_max, args.bin_width):
        upper = lower + args.bin_width
        mask = (snr >= lower) & (snr < upper)
        count = int(mask.sum())
        if count == 0:
            continue

        row: dict[str, float | int | str] = {
            "snr_bin": f"{lower}-{upper}",
            "snr_lower": lower,
            "snr_upper": upper,
            "n": count,
        }
        for target_idx, target in enumerate(TARGETS):
            target_residuals = residuals[mask, target_idx]
            target_absolute_errors = absolute_errors[mask, target_idx]
            row[f"mae_{target}"] = float(np.mean(target_absolute_errors))
            row[f"std_absolute_error_{target}"] = float(
                np.std(target_absolute_errors, ddof=args.ddof)
            )
            row[f"bias_{target}"] = float(np.mean(target_residuals))
            row[f"std_residual_{target}"] = float(
                np.std(target_residuals, ddof=args.ddof)
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    return result


def print_table(result: pd.DataFrame, ddof: int) -> None:
    print(f"\nError statistics by SNR bin (standard deviation ddof={ddof})")
    print(
        f"{'SNR bin':<10} {'n':>6} "
        f"{'Teff MAE +/- std|e| (K)':>25} "
        f"{'FeH MAE +/- std|e|':>23} "
        f"{'logg MAE +/- std|e|':>24}"
    )
    for row in result.itertuples(index=False):
        print(
            f"{row.snr_bin:<10} {row.n:>6d} "
            f"{row.mae_Teff:>10.2f} +/- {row.std_absolute_error_Teff:<8.2f} "
            f"{row.mae_FeH:>8.3f} +/- {row.std_absolute_error_FeH:<7.3f} "
            f"{row.mae_logg:>8.3f} +/- {row.std_absolute_error_logg:<7.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate MAE and error standard deviation in SNR bins."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--snr-data", type=Path, default=DEFAULT_SNR_DATA)
    parser.add_argument(
        "--prediction-scalers",
        type=Path,
        default=None,
        help="Optional scaler NPZ for legacy models.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--snr-min", type=int, default=0)
    parser.add_argument("--snr-max", type=int, default=140)
    parser.add_argument("--bin-width", type=int, default=20)
    parser.add_argument(
        "--ddof",
        type=int,
        default=0,
        help="Delta degrees of freedom for np.std; 0 matches existing repo metrics.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = calculate_stats(args)
    print_table(result, args.ddof)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
