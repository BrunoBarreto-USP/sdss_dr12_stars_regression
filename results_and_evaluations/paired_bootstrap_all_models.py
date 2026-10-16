"""Paired bootstrap MAE comparisons for CNN, DNN, and SpecT versus the paper model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREDICTION_FILES = {
    "Proposta": ROOT / "data" / "paper_hyperparams_test_predictions.npz",
    "DNN": ROOT / "data" / "dnn_test_predictions.npz",
    "CNN": ROOT / "data" / "cnn_test_predictions.npz",
    "SpecT": ROOT / "results_and_evaluations" / "spectral_transformer" / "test_predictions.npz",
}
RESULTS = ROOT / "results_and_evaluations" / "paired_bootstrap_all_models.json"
MESSAGE = ROOT / "results_and_evaluations" / "mensagem_professor_bootstrap_pareado.txt"
TARGETS = ("Teff", "[Fe/H]", "log g")
UNITS = ("K", "dex", "dex")


def bootstrap_paired_differences(differences: np.ndarray, *, n_boot: int = 10000, seed: int = 42):
    """Resample objects jointly for every target and comparator; positive favors the reference."""
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 3 or not len(differences) or not np.isfinite(differences).all() or n_boot < 1:
        raise ValueError('Expected finite object-by-target-by-comparator differences and positive n_boot.')
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, *differences.shape[1:]), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, len(differences), size=len(differences))
        boot[b] = differences[idx].mean(axis=0)
    return differences.mean(axis=0), np.percentile(boot, [2.5, 97.5], axis=0)


def _load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        ids = np.asarray(data["test_ids"])
        names = tuple(np.asarray(data["target_names"]).tolist())
        y_true = np.asarray(data["y_true"], dtype=np.float64)
        y_pred = np.asarray(data["y_pred"], dtype=np.float64)
    if names != TARGETS or ids.ndim != 1 or y_true.shape != (len(ids), 3) or y_pred.shape != y_true.shape:
        raise ValueError(f"Invalid prediction archive shape or target order: {path}")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError(f"Nonfinite predictions or targets: {path}")
    return ids, y_true, y_pred


def _pt(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paper-predictions", type=Path, default=PREDICTION_FILES["Proposta"])
    parser.add_argument("--results-out", type=Path, default=RESULTS)
    parser.add_argument("--message-out", type=Path, default=MESSAGE)
    parser.add_argument("--no-message", action="store_true", help="Write only the scientific JSON report.")
    args = parser.parse_args()
    if args.n_boot < 1:
        parser.error("--n-boot must be positive")

    prediction_files = {**PREDICTION_FILES, "Proposta": args.paper_predictions}
    ids, y_true, paper_pred = _load_predictions(prediction_files["Proposta"])
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Test IDs must be unique for paired comparison.")
    n = len(ids)
    paper_errors = np.abs(y_true - paper_pred)
    baseline_errors = {}
    for name in ("DNN", "CNN", "SpecT"):
        model_ids, model_true, model_pred = _load_predictions(prediction_files[name])
        if not np.array_equal(model_ids, ids) or not np.array_equal(model_true, y_true):
            raise ValueError(f"{name} test IDs or true targets differ from the paper model.")
        baseline_errors[name] = np.abs(y_true - model_pred)

    model_names = tuple(baseline_errors)
    differences = np.stack([baseline_errors[name] - paper_errors for name in model_names], axis=2)
    _, ci = bootstrap_paired_differences(differences,n_boot=args.n_boot,seed=args.seed)

    report = {
        "n_test": n,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "method": "paired percentile bootstrap of per-object absolute-error differences",
        "difference_definition": "MAE_comparator - MAE_paper; positive favors the paper model",
        "prediction_files": {name: str(path) for name, path in prediction_files.items()},
        "paper_mae": {target: float(paper_errors[:, j].mean()) for j, target in enumerate(TARGETS)},
        "comparisons": {},
    }
    n_text = f"{n:,}".replace(",", ".")
    n_boot_text = f"{args.n_boot:,}".replace(",", ".")
    message = [
        "Comparação bootstrap pareada dos MAEs no conjunto de teste. Foram usadas as predições em escala física "
        f"dos mesmos {n_text} objetos (IDs conferidos), com {n_boot_text} reamostragens com reposição (semente {args.seed}).",
        "Defini ΔMAE = MAE do comparador − MAE da rede proposta; valores positivos favorecem a proposta.",
        "",
    ]
    message.append(
        "MAEs da proposta: Teff " + _pt(report["paper_mae"]["Teff"], 2) + " K; "
        + "[Fe/H] " + _pt(report["paper_mae"]["[Fe/H]"], 4) + " dex; "
        + "log g " + _pt(report["paper_mae"]["log g"], 4) + " dex."
    )
    message.append("")

    for model_idx, name in enumerate(model_names):
        report["comparisons"][name] = {}
        message.append(f"{name} (ΔMAE e IC bootstrap de 95%):")
        for target_idx, (target, unit) in enumerate(zip(TARGETS, UNITS)):
            delta = float(differences[:, target_idx, model_idx].mean())
            low = float(ci[0, target_idx, model_idx])
            high = float(ci[1, target_idx, model_idx])
            baseline_mae = float(baseline_errors[name][:, target_idx].mean())
            report["comparisons"][name][target] = {
                "mae_comparator": baseline_mae,
                "mae_paper": report["paper_mae"][target],
                "mae_difference": delta,
                "difference_ci_95": [low, high],
                "ci_excludes_zero": bool(low > 0 or high < 0),
                "unit": unit,
            }
            decimals = 2 if target == "Teff" else 4
            message.append(
                f"• {target}: MAE {_pt(baseline_mae, decimals)} {unit}; "
                f"Δ = {_pt(delta, decimals)} {unit} "
                f"[IC {_pt(low, decimals)}; {_pt(high, decimals)}]."
            )
        message.append("")

    all_positive = all(
        report["comparisons"][name][target]["difference_ci_95"][0] > 0
        for name in model_names for target in TARGETS
    )
    if all_positive:
        message.append(
            "Os nove ICs individuais estão acima de zero: neste test set, a rede proposta teve MAE menor "
            "que DNN, CNN e SpecT nos três parâmetros. Os ICs são percentis bootstrap pareados de 95%, "
            "sem ajuste para múltiplas comparações."
        )
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"JSON: {args.results_out}")
    if not args.no_message:
        args.message_out.parent.mkdir(parents=True, exist_ok=True)
        args.message_out.write_text("\n".join(message) + "\n", encoding="utf-8")
        print(f"Comparison note: {args.message_out}")


if __name__ == "__main__":
    main()
