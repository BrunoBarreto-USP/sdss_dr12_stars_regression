# Stellar parameter estimation from SDSS DR12 spectra

A residual multitask MLP estimates Teff, [Fe/H] and log g from 4,000-pixel stellar
spectra. The shared `stellar_workflow` interface prepares data, trains, evaluates,
and generates the paper figures without pre-existing research files.

## Install and check

Use Python **3.11** on Windows or Linux. From the repository root:

```powershell
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m stellar_workflow smoke --offline
```

The offline smoke test creates synthetic data, trains for one epoch, exports/reloads
a checkpoint, evaluates it, and generates all 12 figures. It needs no credentials,
downloaded dataset, archived model or Torch. Its scores are software checks.
Use `smoke --offline --extended` to also exercise four baselines, all seven ablations
and a one-trial tuning run. Each smoke run chooses a new output directory.

TensorFlow uses CPU on native Windows; use Linux/WSL2 for TensorFlow CUDA.
Install `requirements-torch.txt` only for Transformer training or checkpoint audit.
Install `requirements-notebook.txt` to run the cleared `orchestration.ipynb` guide.

## Train from public data

```powershell
python -m stellar_workflow prepare-data
python -m stellar_workflow train --run-dir outputs/runs/example
python -m stellar_workflow figures --run-dir outputs/runs/example
python -m stellar_workflow audit --run-dir outputs/runs/example
```

The public dataset is pinned to revision
`882320daa2ede7c7b7aca9c7344b9b0e388f5425`. Downloads are anonymous, retried,
checksummed, and completed atomically over ordinary HTTPS. Full Parquet files total
several GB; allow additional space for compact arrays and outputs. `--offline`
requires a verified local cache. All three Parquet files and the compact provenance
manifest are validated before fitting.

For a short real-data round trip, use `python -m stellar_workflow smoke --online`.
For manual sample experiments, use `prepare-data --row-limit 8`, then
`train --run-dir outputs/runs/sample --row-limit 8 --epochs 1 --batch-size 8`.
Sample caches stay separate from full-data caches; never report their scores as paper results.

## Optional stages

```powershell
python -m stellar_workflow baselines --run-dir outputs/runs/example --models ols ridge cnn dnn
python -m stellar_workflow evaluate --run-dir outputs/runs/example --comparators cnn dnn --replace-derived
python -m stellar_workflow ablations --run-dir outputs/runs/example --low-memory
python -m stellar_workflow tune --run-dir outputs/runs/tuned
# Optional Torch, CUDA by default:
python -m stellar_workflow baselines --run-dir outputs/runs/example --models transformer
```

`evaluate` compares only explicitly requested models and fails early for missing or
misaligned archives. Ablations support the paper architecture and the run's matched
epoch/batch budget. Tuning is optional, not required to train the paper architecture.
Use each command's `--help` for overrides. Documented legacy scripts remain adapters
to the same workflow; see [the compatibility guide](scripts/README_single_column.md).

## Reproduce the archived paper

The frozen `paper-v1` experiment is published in a separate Hugging Face **model** repository:
[BrunoBarreto/sdss_dr12_stars_regression](https://huggingface.co/BrunoBarreto/sdss_dr12_stars_regression).
`reference/artifacts.json` pins immutable revision
`3a7b0b12020f577c49d806cd930979f3d5117c27` and the manifest checksum. Public
downloads require no login. An existing local bundle can also be supplied with `--bundle PATH`.

```powershell
python -m stellar_workflow reproduce-paper --run-dir outputs/reference/paper-v1
# Local verified bundle, without a model-repository download:
python -m stellar_workflow reproduce-paper --run-dir outputs/reference/local --bundle PATH
# Optional extra checkpoint inference:
python -m stellar_workflow audit --run-dir outputs/reference/paper-v1 --verify-transformer --output-dir outputs/reference/torch-audit
```

Reproduction verifies archived checkpoints/predictions, downloads the pinned full
dataset, and regenerates bootstrap/ablation/parameter-bin tables and all 12 figures.
It does not retrain or modify the manuscript. Core verification hashes Transformer
artifacts and checks saved predictions; extra checkpoint inference requires Torch.

## Reference experiment

`configs/paper.json` is authoritative: trial 099, seed 42, 80 epochs, batch 256,
128-unit stem, one 32-unit residual block, Teff head (16,), and [Fe/H]/log g heads
(48,32,16): **530,275 parameters**. The archived checkpoint selected epoch 74.
The full split contains **29,422 / 4,846 / 14,443** spectra. Two fixed noisy copies
give 88,266 training examples; Gaussian noise is a regularization perturbation.

| Target | Archived test MAE |
| --- | ---: |
| Teff | 58.717678 K |
| [Fe/H] | 0.101314 dex |
| log g | 0.131862 dex |

Those numbers describe the frozen checkpoint, not a guarantee of bitwise-identical
training on every device. CNN and Li-style DNN are controlled adaptations of the
literature architectures. Bootstrap intervals condition on fitted models and do not
measure seed variation. See [reproducibility and limitations](docs/REPRODUCIBILITY.md).

## Outputs, safety and provenance

Git contains code, configuration, tests, small fixtures and consolidated tuning
metadata. Generated data, models, reports, logs and archives are ignored. Each run
records relative artifact paths, file hashes, dataset fingerprints, configuration,
software versions, source provenance and stage status. Main, baseline and ablation
stages can append to one run; existing training artifacts cannot be overwritten.
Reports and figures live in separate directories. Rerender into a new `--output-dir`,
or explicitly use `--replace-derived`; never use a training directory as derived output.

Source and authored model artifacts use the [MIT license](LICENSE). The public
[dataset](https://huggingface.co/datasets/BrunoBarreto/sdss_dr12_stars_regression)
retains **CC-BY-4.0** and its SDSS/SSPP attribution; see [data attribution](docs/DATA_LICENSE.md).
