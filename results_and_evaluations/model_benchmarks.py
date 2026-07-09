"""Classical benchmarks for comparison with the neural network.

The clean paper release intentionally keeps only the two linear baselines
used in the public comparison workflow: OLS and Ridge. Each method
predicts all three targets at once (multi-output).

All metrics are computed in the unscaled physical space via the fitted
``target_scalers``, matching the project evaluation convention.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score

_MACRO_SCALES: dict[str, float] = {
    "Teff": 100.0,
    "[Fe/H]": 0.20,
    "log g": 0.20,
}


def _unscale(y_scaled: np.ndarray, target_scalers: Sequence) -> np.ndarray:
    """Inverse-transform ``(N, n_targets)`` scaled targets to physical space."""
    y = np.empty_like(y_scaled, dtype=np.float64)
    for idx, scaler in enumerate(target_scalers):
        y[:, idx] = scaler.inverse_transform(y_scaled[:, idx].reshape(-1, 1)).ravel()
    return y


def _compute_metrics(
    y_pred_scaled: np.ndarray,
    y_true_scaled: np.ndarray,
    target_scalers: Sequence,
    target_names: Sequence[str],
) -> dict[str, float]:
    """Per-target MAE, R2, and macro MAE in unscaled space."""
    y_pred = _unscale(y_pred_scaled, target_scalers)
    y_true = _unscale(y_true_scaled, target_scalers)

    metrics: dict[str, float] = {}
    macro_sum = 0.0
    for idx, name in enumerate(target_names):
        mae = mean_absolute_error(y_true[:, idx], y_pred[:, idx])
        r2 = r2_score(y_true[:, idx], y_pred[:, idx])
        metrics[f"MAE_{name}"] = mae
        metrics[f"R2_{name}"] = r2
        macro_sum += mae / _MACRO_SCALES.get(name, 1.0)
    metrics["macro_mae"] = macro_sum / len(target_names)
    return metrics


def _make_row(
    method: str,
    mode: str,
    metrics: dict[str, float],
    fit_time: float,
) -> dict[str, object]:
    return {"method": method, "mode": mode, **metrics, "fit_time_s": round(fit_time, 2)}


def _show_progress(results: list[dict[str, object]]) -> None:
    """Print the current benchmark summary as a compact text table."""
    if not results:
        return

    mae_keys = [key for key in results[0] if key.startswith("MAE_")]
    header = f"  {'Method':<28}"
    for key in mae_keys:
        header += f" {key:>12}"
    header += f" {'macro_mae':>10} {'time':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in results:
        line = f"  {row['method']:<28}"
        for key in mae_keys:
            line += f" {row[key]:>12.3f}"
        line += f" {row['macro_mae']:>10.4f} {row['fit_time_s']:>6.1f}s"
        print(line)


def _fit_evaluate_multi(
    model,
    method_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scalers: Sequence,
    target_names: Sequence[str],
) -> dict[str, object]:
    """Fit one multi-output model and evaluate it on the test split."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

    predictions = model.predict(X_test)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    metrics = _compute_metrics(predictions, y_test, target_scalers, target_names)
    return _make_row(method_name, "multi-output", metrics, fit_time)


def run_linear_benchmarks(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scalers: Sequence,
    target_names: Sequence[str],
    _prior_results: list[dict] | None = None,
) -> list[dict[str, object]]:
    """Fit the public-release linear baselines and return result dicts."""
    del X_val, y_val

    results: list[dict[str, object]] = list(_prior_results or [])
    args = (X_train, y_train, X_test, y_test, target_scalers, target_names)

    print("  OLS ...")
    results.append(_fit_evaluate_multi(LinearRegression(), "OLS", *args))
    _show_progress(results)

    print("  Ridge ...")
    alphas = np.logspace(-3, 3, 13)
    results.append(_fit_evaluate_multi(RidgeCV(alphas=alphas), "Ridge", *args))
    _show_progress(results)

    return results


def add_external_result(
    results_df: pd.DataFrame,
    method_name: str,
    predictions_scaled: np.ndarray,
    y_test: np.ndarray,
    target_scalers: Sequence,
    target_names: Sequence[str],
    *,
    mode: str = "multi-output",
    fit_time: float = float("nan"),
) -> pd.DataFrame:
    """Inject results from an external model into the benchmark table."""
    metrics = _compute_metrics(predictions_scaled, y_test, target_scalers, target_names)
    row = _make_row(method_name, mode, metrics, fit_time)
    return pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)


def run_all_benchmarks(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scalers: Sequence,
    target_names: Sequence[str],
) -> pd.DataFrame:
    """Run the public-release benchmark set and return a results DataFrame."""
    print("=" * 60)
    print("Running classical benchmarks: OLS and Ridge")
    print("=" * 60)
    results = run_linear_benchmarks(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        target_scalers,
        target_names,
    )
    return pd.DataFrame(results)


def print_benchmark_table(df: pd.DataFrame) -> None:
    """Print a neatly formatted benchmark results table."""
    mae_cols = [column for column in df.columns if column.startswith("MAE_")]

    header = f"{'Method':<28}"
    for column in mae_cols:
        header += f" {column:>12}"
    header += f" {'macro_mae':>10} {'time':>7}"

    print()
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, row in df.iterrows():
        line = f"  {row['method']:<28}"
        for column in mae_cols:
            line += f" {row[column]:>12.3f}"
        line += f" {row['macro_mae']:>10.4f} {row['fit_time_s']:>6.1f}s"
        print(line)
