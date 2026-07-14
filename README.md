# Multi-Variable Stellar Parameter Estimation Using Residual Multitask Neural Networks

This repository reproduces the workflow used in the paper by Bruno Santos Meneses Barreto and Marcio Eisencraft for estimating `Teff`, `[Fe/H]`, and `log g` from SDSS DR12 stellar spectra with a compact residual multitask MLP.

The code covers the full pipeline:

- download the public benchmark splits from Hugging Face,
- build the compact preprocessed NPZ used for training,
- run the Bayesian hyperparameter search,
- train the selected paper model,
- generate the main evaluation figure and test metrics,
- optionally compare the paper model with OLS, Ridge, a StarNet-style CNN, and a Li-style DNN baseline.

## Paper Workflow

The notebook follows the same sequence described in the paper:

1. load the SDSS DR12 benchmark split (`30k` train, `5k` validation, `15k` test),
2. resample each spectrum to a common 4000-point log-wavelength grid,
3. shift spectra to the stellar rest frame,
4. standardize each spectrum individually and scale targets with `RobustScaler`,
5. optionally compare against `OLS`, `Ridge`, a StarNet-style CNN, and a Li-style DNN,
6. tune and train the residual multitask neural network,
7. report scatter plots, MAE, and SNR-binned error statistics.

The saved paper-model metrics in this checkout are approximately:

- `Teff` MAE: `59.5 K`
- `[Fe/H]` MAE: `0.101 dex`
- `log g` MAE: `0.130 dex`
- trainable parameters: `542,771`

The CNN and Li-style DNN are controlled comparison baselines: they use the
same compact processed spectra, split, and optional augmentation as the paper
model. They are not exact reproductions of the original external papers.

## Main Entry Point

Use [orchestration.ipynb](/C:/Users/barre/Desktop/github_submission_clean/github_submission_clean/orchestration.ipynb) as the main entry point. It is the only README-guided workflow in this repository and is organized to mirror the paper sections.

## Repository Structure

```text
orchestration.ipynb          End-to-end notebook for the paper workflow
data_acquisition/            Hugging Face download and split-loading helpers
data_preprocessing/          Spectrum preprocessing and compact NPZ generation
data_exploration/            Figures and exploratory summaries used in the paper
model_definitions/           Residual multitask MLP and tuner builders
training/                    Hyperparameter search and fixed-model training
results_and_evaluations/     Metrics, baselines, scatter plots, and SNR analysis
data/                        Local dataset cache and compact benchmark files
models/                      Saved trained models and weights
figs/                        Generated figures
```

## Dataset

The code uses the public Hugging Face dataset:

- `BrunoBarreto/sdss_dr12_stars_regression`

Files are downloaded automatically into `data/` when required.

## Setup

Python `3.10` or `3.11` is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## What Gets Generated

Running the notebook can create these local artifacts:

- `data/train.parquet`, `data/validation.parquet`, `data/test.parquet`
- `data/sdss_dr12_processed_flux_benchmark.npz`
- `models/paper_hyperparams_model.keras`
- `models/paper_hyperparams_best.weights.h5`
- `figs/paper_model_reference_vs_predicted.png`
- `results_and_evaluations/paper_hyperparams_metrics.json`
- `results_and_evaluations/paper_model_test_metrics.json`
- `results_and_evaluations/paper_model_snr_error_stats.csv`
- `results_and_evaluations/classical_benchmark_results.csv`

These outputs are local working artifacts and should generally stay out of Git.
