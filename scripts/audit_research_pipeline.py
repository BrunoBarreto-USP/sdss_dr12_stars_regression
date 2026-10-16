"""Audit the local data, selected checkpoints and reported research artifacts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import zipfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_preprocessing.validation import validate_compact_dataset
from data_preprocessing.dataset import normalize_flux_per_spectrum
from results_and_evaluations.prediction_utils import OUTPUT_NAMES, prediction_matrix

TARGETS = ("Teff", "[Fe/H]", "log g")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def cached_feature_hash(path, name):
    with zipfile.ZipFile(path) as archive, archive.open(name + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        reader = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, fortran, dtype = reader(stream)
        if len(shape) != 2 or shape[1] != 4000 or fortran or dtype != np.float32:
            raise ValueError(
                f"Unexpected cached feature layout for {name}: {shape}, {dtype}."
            )
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return shape, hasher.hexdigest()


def audit_features(path, *, normalize):
    hasher = hashlib.sha256()
    count, max_mean, max_std_error = 0, 0.0, 0.0
    first = None
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=256, columns=["processed_flux"]
    ):
        x = (
            batch.column(0)
            .flatten()
            .to_numpy(zero_copy_only=False)
            .astype(np.float32, copy=False)
        )
        x = x.reshape(batch.num_rows, 4000)
        if normalize:
            x = normalize_flux_per_spectrum(x)
        if not np.isfinite(x).all():
            raise ValueError(f"Nonfinite processed inputs in {path}.")
        means, stds = x.mean(axis=1, dtype=np.float64), x.std(axis=1, dtype=np.float64)
        max_mean = max(max_mean, float(abs(means).max()))
        max_std_error = max(max_std_error, float(abs(stds - 1).max()))
        hasher.update(np.ascontiguousarray(x).tobytes())
        if first is None:
            first = x[:64].copy()
        count += len(x)
    if max_mean > 1e-4 or max_std_error > 1e-4:
        raise ValueError(
            f"Unexpected per-spectrum standardization in {path}: {max_mean}, {max_std_error}."
        )
    return {
        "rows": count,
        "sha256": hasher.hexdigest(),
        "max_abs_mean": max_mean,
        "max_abs_std_minus_one": max_std_error,
    }, first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "outputs/paper_figures/single_column/corrected_best_trial",
    )
    parser.add_argument("--paper-dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir
    tests = subprocess.run(
        [sys.executable, "-u", "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (run / "research_regression_tests.log").write_text(tests.stdout, encoding="utf-8")
    if tests.returncode:
        raise RuntimeError(
            "Regression tests failed; inspect research_regression_tests.log before auditing artifacts."
        )
    source_folders = (
        "data_acquisition",
        "data_preprocessing",
        "data_exploration",
        "model_definitions",
        "training",
        "results_and_evaluations",
        "finetuning",
        "scripts",
        "tests",
    )
    sources = [p for folder in source_folders for p in (ROOT / folder).glob("*.py")]
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    report = {
        "python": platform.python_version(),
        "python_sources_parsed": len(sources),
        "source_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in sources},
    }
    compact = ROOT / "data/sdss_dr12_processed_flux_benchmark.npz"
    with np.load(compact, allow_pickle=False) as data:
        report["split_counts"] = validate_compact_dataset(data, ROOT / "data")
        normalize = bool(data["flux_normalized_per_spectrum"])
        center = data["label_robust_center"].astype(np.float64)
        scale = data["label_robust_scale"].astype(np.float64)
        report["train_only_target_scaler_verified"] = True
        features = {}
        first_test = None
        for split, short in (
            ("train", "train"),
            ("validation", "val"),
            ("test", "test"),
        ):
            shape, fingerprint = cached_feature_hash(compact, f"X_{short}_features")
            detail, first = audit_features(
                ROOT / "data" / f"{split}.parquet", normalize=normalize
            )
            if (
                tuple(shape) != (report["split_counts"][split], 4000)
                or detail["sha256"] != fingerprint
            ):
                raise ValueError(
                    f"Compact {split} inputs differ from the Parquet-derived inputs."
                )
            features[split] = detail
            if split == "test":
                first_test = first
        report["full_processed_input_audit"] = features
        y_train = np.asarray(data["y_train_targets"], dtype=np.float64)
    print("Full feature, ID, label and training-only scaler audit passed.", flush=True)

    config = json.loads((run / "run_config.json").read_text())
    history = pd.read_csv(run / "history.csv")
    assert len(history) == config["epochs"] == 80
    assert int(history.val_normalized_mae_sum.argmin()) + 1 == config["selected_epoch"]
    std = np.std(y_train, axis=0)
    inverse_variance = 3 / std**2 / np.sum(1 / std**2)
    np.testing.assert_allclose(
        inverse_variance, [config["loss_weights"][k] for k in OUTPUT_NAMES], rtol=1e-12
    )
    with np.load(run / "test_predictions.npz", allow_pickle=False) as z:
        ids, true, pred = z["test_ids"], z["y_true"], z["y_pred"]
        assert tuple(z["target_names"]) == TARGETS
    assert len(np.unique(ids)) == len(ids) == report["split_counts"]["test"]
    np.testing.assert_allclose(
        np.mean(abs(pred - true), axis=0),
        [config["test_mae"][t] for t in TARGETS],
        rtol=1e-12,
    )
    paired = json.loads((run / "paired_bootstrap.json").read_text())
    for baseline in ("CNN", "DNN", "SpecT"):
        with np.load(paired["prediction_files"][baseline], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z["test_ids"])
            np.testing.assert_array_equal(true, z["y_true"])
            delta = np.mean(abs(z["y_pred"] - true) - abs(pred - true), axis=0)
        np.testing.assert_allclose(
            delta,
            [paired["comparisons"][baseline][t]["mae_difference"] for t in TARGETS],
            rtol=1e-12,
        )
    ranges = pd.read_csv(run / "parameter_range_errors.csv")
    for j, target in enumerate(TARGETS):
        rows = ranges[ranges.target == target]
        assert rows.n.sum() == len(ids)
        np.testing.assert_allclose(
            np.average(rows.mae, weights=rows.n),
            abs(pred[:, j] - true[:, j]).mean(),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            np.average(rows.bias, weights=rows.n),
            (pred[:, j] - true[:, j]).mean(),
            rtol=1e-12,
        )
    trials_checked = 0
    for path in (
        ROOT / "results_and_evaluations/keras_tuner/stellar_baseline_orchestration"
    ).glob("trial_*/trial.json"):
        trial = json.loads(path.read_text())
        metrics = trial["metrics"]["metrics"]
        if trial["status"] != "COMPLETED":
            continue
        values = [
            metrics[f"val_{k}_mae"]["observations"][0]["value"][0] for k in OUTPUT_NAMES
        ]
        np.testing.assert_allclose(
            np.sum(values / std), trial["score"], atol=1e-12, rtol=1e-12
        )
        trials_checked += 1
    report["archived_trial_selection_scores_verified"] = trials_checked
    print(
        "Checkpoint selection, target weighting, paired comparisons and parameter-bin summaries passed.",
        flush=True,
    )

    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    model = tf.keras.models.load_model(
        run / "retrained_paper_model.keras", compile=False
    )
    assert model.count_params() == 530275
    forward = (
        prediction_matrix(model(first_test, training=False)).astype(np.float64) * scale
        + center
    )
    np.testing.assert_allclose(forward, pred[: len(forward)], atol=0.002, rtol=1e-6)
    audit = json.loads((run / "architecture_audit.json").read_text())
    assert digest(run / "retrained_paper_model.keras") == audit["model_sha256"]
    assert digest(run / "test_predictions.npz") == audit["predictions_sha256"]
    report["direct_checkpoint_inference_objects"] = len(forward)
    report["trainable_parameters"] = model.count_params()
    report["model_sha256"] = audit["model_sha256"]
    report["legacy_paper_model_preserved"] = (
        digest(ROOT / "models/paper_hyperparams_model.keras")
        == audit["previous_paper_model_sha256"]
    )
    assert report["legacy_paper_model_preserved"]
    report["selected_epoch"] = config["selected_epoch"]
    report["test_mae"] = config["test_mae"]
    from finetuning.checkpoint_io import load_checkpoint

    checkpoint = load_checkpoint(
        ROOT / "results_and_evaluations/spectral_transformer/best_model.pt"
    )
    report["transformer_cross_platform_checkpoint_load"] = True
    report["transformer_selected_epoch"] = checkpoint["epoch"]

    ablations = {}
    for folder in sorted((run / "ablations").iterdir()):
        if not (folder / "metrics.json").exists():
            continue
        info = json.loads((folder / "metrics.json").read_text())
        h = pd.read_csv(folder / "history.csv")
        assert (
            len(h) == 80
            and int(h.val_normalized_mae_sum.argmin()) + 1 == info["selected_epoch"]
        )
        with np.load(folder / "test_predictions.npz", allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z["test_ids"])
            for j, target in enumerate(z["target_names"]):
                source_index = TARGETS.index(target)
                np.testing.assert_array_equal(true[:, source_index], z["y_true"][:, j])
                np.testing.assert_allclose(
                    np.mean(abs(z["y_pred"][:, j] - z["y_true"][:, j])),
                    info["test_mae"][target],
                    rtol=1e-12,
                )
        for target, widths in info["heads"].items():
            assert widths == config["heads"][target]
        ablations[folder.name] = {
            "selected_epoch": info["selected_epoch"],
            "params": info["params"],
            "mae": info["test_mae"],
        }
    report["verified_ablations"] = ablations
    expected = {
        "no_residual_no_layernorm",
        "single_teff",
        "single_feh",
        "single_logg",
        "no_skip",
        "no_layernorm",
        "no_noise_augmentation",
    }
    assert set(ablations) == expected
    ablation_intervals = json.loads(
        (run / "ablations/ablation_paired_bootstrap.json").read_text()
    )
    assert (
        ablation_intervals["n_test"] == len(ids)
        and ablation_intervals["n_boot"] == 10000
    )
    for name, targets in ablation_intervals["comparisons"].items():
        if name == "no_multitask":
            alternative = np.column_stack(
                [
                    np.load(run / "ablations" / part / "test_predictions.npz")[
                        "y_pred"
                    ][:, 0]
                    for part in ("single_teff", "single_feh", "single_logg")
                ]
            )
        else:
            alternative = np.load(run / "ablations" / name / "test_predictions.npz")[
                "y_pred"
            ]
        differences = np.mean(abs(alternative - true) - abs(pred - true), axis=0)
        for j, target in enumerate(TARGETS):
            np.testing.assert_allclose(
                differences[j], targets[target]["difference"], rtol=1e-12
            )
            interval = targets[target]["ci_95"]
            assert (
                len(interval) == 2
                and np.isfinite(interval).all()
                and interval[0] <= interval[1]
            )
    report["ablation_bootstrap_comparisons"] = sum(
        len(targets) for targets in ablation_intervals["comparisons"].values()
    )
    log = (run / "research_regression_tests.log").read_text(errors="replace")
    match = re.search(r"Ran (\d+) tests", log)
    if not match or not re.search(r"\nOK\s*$", log):
        raise ValueError(
            "Run the regression test suite successfully before auditing final artifacts."
        )
    report["regression_tests_passed"] = int(match.group(1))
    report["scope_limits"] = [
        "Local fixed-checkpoint reproduction, not a multi-seed or external-survey validation.",
        "Original acquisition and physical-object identifiers cannot be rechecked without the SSPP FITS catalogue.",
        "The saved Transformer was loaded and its predictions audited; GPU training was not repeated.",
    ]
    report["status"] = "passed"
    (run / "research_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    notes = f"""# Revisão dos scripts da pesquisa

Auditoria local concluída: {report["regression_tests_passed"]} testes de regressão passaram;
{report["python_sources_parsed"]} fontes Python foram analisadas sintaticamente.
Os três arquivos de entradas processadas foram comparados integralmente, por SHA-256,
entre Parquet e NPZ: 29.422 objetos de treino, 4.846 de validação e 14.443 de teste.
Os valores são finitos, a padronização por espectro foi conferida e os alvos/escaladores
correspondem às divisões locais, com ajuste do scaler somente no treino.

## Problemas corrigidos

- `training/run_paper_ablations.py` montava cabeças antigas. Agora deriva as larguras dos
  mesmos templates do trial vencedor; o modelo completo tem 530.275 parâmetros.
- Entradas genéricas de treino usavam pesos inversos ao desvio padrão, enquanto o
  experimento do artigo usa variância inversa. O cálculo foi centralizado e alinhado.
  A normalização por desvio padrão do critério de validação continua separada.
- Caches compactos podiam ser reutilizados após alterações dos splits. A validação agora
  compara IDs, rótulos, medianas e IQRs com os Parquets antes do treino.
- O cache de aumento de dados não identificava sua fonte. Agora valida forma, dtype,
  receita e hash dos valores originais. Caches anteriores são comparados aos valores esperados.
- `hf_data.py` podia produzir uma exceção de variável não inicializada ao informar colunas
  ausentes. O erro agora identifica corretamente o arquivo e as colunas.
- A leitura de checkpoints do Transformer dependia do sistema operacional. O novo loader
  trata metadados de caminhos Windows/POSIX sem alterar globalmente o módulo pathlib.
- Utilitários de inferência e SNR agora tratam saídas nomeadas na ordem canônica dos alvos.
  O softplus do utilitário histórico de incerteza foi tornado numericamente estável.
- Treinos e ablações recusam sobrescrever resultados existentes nos caminhos protegidos.
- README, fluxo ativo, dependências diretas e instruções foram alinhados ao experimento atual.
  A nota bootstrap deixou de incluir uma recomendação desatualizada sobre FLOPs de slides.

## Conferência dos resultados

- Arquitetura: trial 099, 530.275 parâmetros; checkpoint da época {config["selected_epoch"]} de 80.
- Reproduzidas as pontuações de seleção dos {report["archived_trial_selection_scores_verified"]} trials arquivados.
- Inferência direta conferida em {report["direct_checkpoint_inference_objects"]} espectros contra as previsões salvas.
- Comparações pareadas: IDs e rótulos físicos idênticos para CNN, DNN, Transformer e proposta.
- MAEs do modelo principal preservados: 58,717678 K; 0,101314 dex; 0,131862 dex.
- Os intervalos por parâmetro incluem todos os objetos e recuperam os MAEs globais quando
  ponderados pelas contagens. Os vieses usam previsão menos referência SSPP.
- {len(ablations)} redes de ablação verificadas: histórico completo, seleção por validação,
  cabeças, IDs, rótulos e MAEs recomputados. O controle sem ruído mantém 344 passos por época.
- Modelos e resultados históricos permanecem preservados. Os resultados atuais têm caminhos próprios.

## Organização e limites

A organização por aquisição, preparação, definição, treino e avaliação é adequada.
O fluxo ativo está descrito em `scripts/README_single_column.md`; os scripts auxiliares e o
notebook mantêm nomes históricos e não devem ser usados para identificar o checkpoint atual.
O JSON de auditoria registra hashes dos códigos e artefatos; `software_versions.json` registra
as bibliotecas locais. Os testes de regressão cobrem os contratos que poderiam alterar alvos,
arquitetura, regularização ou seleção do checkpoint.

Esta revisão valida o fluxo local e os checkpoints arquivados. Não é uma garantia de ausência
de qualquer defeito: o download original e os identificadores físicos dos objetos exigem o
catálogo SSPP FITS, indisponível nesta pasta. O treino em GPU do Transformer não foi repetido;
seu checkpoint foi carregado e as previsões de teste foram conferidas. Os resultados usam uma
semente de treino e não constituem validação externa ou prova de desempenho uniforme nos extremos.
As comparações usam perdas e orçamentos de busca próprios de cada arquitetura, explicitados no artigo.
"""
    (run / "research_audit.md").write_text(notes, encoding="utf-8")
    (ROOT / "RESEARCH_CODE_REVIEW.md").write_text(notes, encoding="utf-8")
    if args.paper_dir:
        import shutil

        shutil.copy2(run / "research_audit.json", args.paper_dir / "reproducibility")
        shutil.copy2(run / "research_audit.md", args.paper_dir / "reproducibility")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("source_sha256", "full_processed_input_audit")
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
