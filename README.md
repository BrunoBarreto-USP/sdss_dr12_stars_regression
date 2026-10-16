# Compact Residual Multitask Neural Networks for Stellar Parameter Estimation

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

## Current article experiment

The active experiment uses the object-deduplicated local splits: **29,422 training, 4,846
validation and 14,443 test spectra**. Inputs contain 4000 wavelength pixels. Target scalers
are fitted on training labels only. Two fixed noisy copies increase the training inputs to
88,266; these Gaussian perturbations are regularization, with labels unchanged.

The architecture is exactly the best saved Bayesian trial, `trial_099`: a 128-unit stem,
one 32-unit residual block, a `Teff=(16,)` head and `[Fe/H]=(48,32,16)` and
`log g=(48,32,16)` heads. It has **530,275 trainable parameters**. An 80-epoch run with seed 42
selected epoch 74 by normalized validation MAE. Curves, weights, predictions, bootstrap intervals
and the article tables all use that same checkpoint.

| Target | Test MAE | Individual bootstrap 95% interval |
| --- | ---: | ---: |
| Teff | 58.717678 K | 57.433322–60.075142 K |
| [Fe/H] | 0.101314 dex | 0.099023–0.103645 dex |
| log g | 0.131862 dex | 0.128282–0.135533 dex |

The model requires approximately 1.064 million forward-pass FLOPs per spectrum, excluding
preprocessing. It uses about 96% fewer parameters and estimated FLOPs than the three-network
Li-style DNN, while reducing temperature MAE by 1.59 K. Performance is not uniform across the
retained ranges: the parameter-bin diagnostics explicitly report errors, signed bias and counts.

Bootstrap reports use 10,000 resamples of the same test objects. Paired intervals compare fixed
checkpoints and do not measure training-seed variation. Earlier experiments remain preserved and
are not the source of the current article's metrics.

The CNN and Li-style DNN are controlled comparison baselines: they use the
same compact processed spectra, split, and optional augmentation as the paper
model. They are not exact reproductions of the original external papers.

## Reproduce the current article

Use [scripts/README_single_column.md](scripts/README_single_column.md) for the active commands.
The consistent experiment is stored in `outputs/paper_figures/single_column/corrected_best_trial/`.

For a fresh training, choose a new output folder; existing histories are protected:

```powershell
python scripts/retrain_paper_with_history.py --out-dir outputs/paper_reproduction --epochs 80 --threads 6
python scripts/finalize_corrected_experiment.py --run-dir outputs/paper_reproduction
python scripts/generate_paper_figures.py --run-dir outputs/paper_reproduction
python scripts/analyze_parameter_ranges.py --run-dir outputs/paper_reproduction
```

The archived comparisons require their saved baseline predictions and the exact local dataset
splits. The compact NPZ is checked against Parquet IDs, labels and training-only scaler values
before use. Downloading an older public snapshot does not replace object deduplication.

`orchestration.ipynb` remains available as the broader exploratory workflow. The commands above
define the current article experiment; historical output names in the notebook refer to earlier runs.

Run the regression checks and artifact audit with:

```powershell
python -m unittest discover -s tests -v
python scripts/audit_research_pipeline.py
```

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

Python `3.11` is the locally verified interpreter. The audit records installed library versions.

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
- `outputs/paper_figures/single_column/corrected_best_trial/retrained_paper_model.keras`
- `outputs/paper_figures/single_column/corrected_best_trial/best.weights.h5`
- `models/benchmarks/fabbro_cnn.keras` and `models/benchmarks/li_dnn_*.keras`
- `outputs/paper_figures/single_column/*_single_column.pdf` and matching PNGs
- `outputs/paper_figures/single_column/corrected_best_trial/run_config.json`
- `outputs/paper_figures/single_column/corrected_best_trial/paired_bootstrap.json`
- `outputs/paper_figures/single_column/corrected_best_trial/parameter_range_errors.csv`
- `results_and_evaluations/classical_benchmark_results.csv`
- `outputs/paper_figures/single_column/corrected_best_trial/ablations/ablation_summary.csv`

These outputs are local working artifacts and should generally stay out of Git.
