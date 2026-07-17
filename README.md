# Multi-Variable Stellar Parameter Estimation Using Residual Multitask Neural Networks

This repository reproduces the workflow used in the paper by Bruno Santos Meneses Barreto and Marcio Eisencraft for estimating `Teff`, `[Fe/H]`, and `log g` from SDSS DR12 stellar spectra with a compact residual multitask MLP.

The code covers the full pipeline:

- download the public benchmark splits from Hugging Face,
- build the compact preprocessed NPZ used for training,
- run the Bayesian hyperparameter search,
- train the selected paper model,
- generate the main evaluation figure and test metrics,
- optionally compare the paper model with OLS, Ridge, a StarNet-style CNN, and a Li-style DNN baseline.

An optional GPU-only PyTorch spectral Transformer is also provided for a
larger attention-based comparison. It is not part of the default notebook run.

## Paper Workflow

The notebook follows the same sequence described in the paper:

1. load the SDSS DR12 benchmark split (`30k` train, `5k` validation, `15k` test),
2. use the released 4000-point processed spectral representation,
3. build a compact NPZ and scale targets with `RobustScaler`,
4. optionally compare against `OLS`, `Ridge`, a StarNet-style CNN, and a Li-style DNN,
5. tune and train the residual multitask neural network,
6. run the residual, layer-normalization, and multitask ablations,
7. report scatter plots, MAE, and SNR-binned error statistics.

The paper-model metrics reported for this workflow are approximately:

- `Teff` MAE: `59.5 K`
- `[Fe/H]` MAE: `0.101 dex`
- `log g` MAE: `0.130 dex`
- trainable parameters: `542,771`

The CNN and Li-style DNN are controlled comparison baselines: they use the
same compact processed spectra, split, and optional augmentation as the paper
model. They are not exact reproductions of the original external papers.

## Main Entry Point

Use [orchestration.ipynb](orchestration.ipynb) as the main entry point. It is the only README-guided workflow in this repository and is organized to mirror the paper sections.

## Repository Structure

```text
orchestration.ipynb          End-to-end notebook for the paper workflow
data_acquisition/            Hugging Face download and split-loading helpers
data_preprocessing/          Spectrum preprocessing and compact NPZ generation
data_exploration/            Figures and exploratory summaries used in the paper
model_definitions/           Residual multitask MLP and tuner builders
training/                    Hyperparameter search and fixed-model training
results_and_evaluations/     Metrics, baselines, scatter plots, and SNR analysis
finetuning/                  Optional GPU-only PyTorch spectral Transformer
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

## Optional Spectral Transformer

`finetuning/train_spectral_transformer.py` implements a 1-D ViT-style
regressor with 16-pixel wavelength patches, a CLS token, learned positional
embeddings, and six Transformer encoder layers (`d_model=256`, 8 heads).

Training this model on CPU is impractically slow. Use a CUDA-enabled PyTorch
installation and a GPU; CPU execution is blocked by default and is available
only through `--allow-cpu` for a small debugging run.

Install the PyTorch build matching the CUDA version of the training machine,
then run:

```powershell
python .\finetuning\train_spectral_transformer.py
```

Architecturally, this is closest to the Vision Transformer of
[Dosovitskiy et al. (2021)](https://arxiv.org/abs/2010.11929), built on the
Transformer encoder of [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762).
The astronomy-specific [Spectral Transformer (SPT)](https://doi.org/10.1051/0004-6361/202347994)
is a relevant domain reference, but it uses a different attention mechanism
and predicts red-giant age and mass rather than these three atmospheric labels.

## What Gets Generated

Running the notebook can create these local artifacts:

- `data/train.parquet`, `data/validation.parquet`, `data/test.parquet`
- `data/sdss_dr12_processed_flux_benchmark.npz`
- `models/paper_hyperparams_model.keras`
- `models/paper_hyperparams_best.weights.h5`
- `models/benchmarks/fabbro_cnn.keras` and `models/benchmarks/li_dnn_*.keras`
- `figs/paper_model_reference_vs_predicted.png`
- `results_and_evaluations/paper_hyperparams_metrics.json`
- `results_and_evaluations/paper_model_test_metrics.json`
- `results_and_evaluations/paper_model_snr_error_stats.csv`
- `results_and_evaluations/classical_benchmark_results.csv`
- `results_and_evaluations/paper_ablations/ablation_summary.csv`

These outputs are local working artifacts and should generally stay out of Git.
