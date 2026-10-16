# Single-column article experiment

The active experiment uses exactly the architecture in
`results_and_evaluations/keras_tuner/stellar_baseline_orchestration/trial_099/trial.json`:
a 128-unit stem, one 32-unit residual block, heads Teff=(16), [Fe/H]=(48,32,16),
log g=(48,32,16), and 530,275 trainable parameters. Predictions, weight diagnostics,
learning curves, confidence intervals and complexity refer to one validation-selected checkpoint.

## Corrected training and evaluation

Run from the ML project root:

```powershell
python scripts/retrain_paper_with_history.py --epochs 80 --threads 6 --out-dir outputs/paper_reproduction
python scripts/finalize_corrected_experiment.py --run-dir outputs/paper_reproduction
python scripts/retrain_corrected_ablations.py --run-dir outputs/paper_reproduction --threads 6 --low-memory
python scripts/retrain_corrected_ablations.py --run-dir outputs/paper_reproduction --summarize-only
python scripts/generate_paper_figures.py --run-dir outputs/paper_reproduction --paper-dir "C:\Users\barre\Documents\els-cas-templates\els-cas-templates"
```

The completed experiment is under `outputs/paper_figures/single_column/corrected_best_trial/`.
An existing history is never overwritten: supply a fresh `--out-dir` for another training and the
same directory through `--run-dir` to evaluation and figure generation. Matched-head ablations
are stored separately under `corrected_best_trial/ablations/`; those histories are preserved too.
For limited RAM, `--low-memory` shuffles sample indices into disk-backed offline copies.
The augmented values and initial sample order were verified to match the in-memory implementation exactly.
The cache stores a hash of the source training values and refuses mismatched inputs.

Seven trained ablation networks cover: removing skip connections only, removing layer normalization
only, removing both, three independent single-target networks, and a no-noise control. The no-noise
control repeats each original spectrum three times without perturbation, retaining the same sample
order, 88,266 presentations per epoch and 344 optimizer steps. It isolates the added noise from the
number of training steps. All ablations use the selected heads and validation-only checkpoints.
Their paired-bootstrap report compares fixed checkpoints, with 10,000 joint object resamples.

For the existing completed experiment, omit `--run-dir` to use `corrected_best_trial` in reporting
and figure commands. Do not retrain into that directory: its histories are protected.

Training uses seed 42, 80 epochs, batch size 256, two offline Gaussian-noise copies, AdamW,
5% warm-up and cosine decay. The dedicated command explicitly uses inverse-variance Huber weights
normalized to sum three. Checkpoints minimize the sum of robust-scaled validation MAEs divided
by training target standard deviations. Hyperparameters and actual head widths are checked
against the best recorded trial before training.

Finalization reuses the existing `bootstrap_mae` and `paired_bootstrap_all_models` implementations.
It verifies all 14,443 test IDs and physical labels, checks predictions against direct checkpoint
inference, computes 10,000 percentile bootstrap resamples (seed 42), and profiles batch-one FLOPs.
Paired differences are baseline MAE minus proposed MAE. Intervals describe test-sample uncertainty
for fixed checkpoints, without multiplicity adjustment; they do not measure training-seed variability.

The full figure command also computes `parameter_range_errors.csv/json` and its three-column
figure: MAE, signed residual and test counts over the full retained reference ranges. The standalone
`analyze_parameter_ranges.py` command updates that diagnostic alone. Bins with fewer than 30 objects
are marked; each target's counts sum to 14,443 and its weighted bin MAEs recover the global MAE.

Run `python -m unittest discover -s tests -v` for regression checks and
`python scripts/audit_research_pipeline.py` for full data/cache hashes, training-only scaler checks,
prediction alignment, direct checkpoint inference, trial-score reproduction and ablation audits.
The latter expects a completed reference experiment, runs the regression suite and writes
`corrected_best_trial/research_regression_tests.log`. See `RESEARCH_CODE_REVIEW.md` for findings
and the limits of the local review.

## Figure style and data

Figures use PDFs with embedded STIXGeneral/STIX fonts and 600-dpi PNGs, exported to
`outputs/paper_figures/single_column/`. Wavelengths follow the actual logarithmic grid.
The flux-scale distribution, Bayesian-optimization curve and prediction scatter are 150 mm wide;
the remaining figures are 166 mm wide. A complete model schematic reads layer widths and head
activations from the evaluated Keras checkpoint and accompanies the residual-block schematic.
All test examples are retained. Prediction points have area 1.1 pt^2, alpha=0.25 and ascending-SNR
drawing order. The sky map uses alpha=0.35. Color bars remain opaque. Style and provenance are
recorded in `manifest.json`.

The sky map uses a 32-neighbour spherical-cap density estimate and logarithmic color scale.
The first-layer matrix uses the diverging berlin palette with a dark neutral centre and a
symmetric-logarithmic scale without clipping. Its linear threshold is the 25th percentile of
absolute weights. A larger lower row shows the full normalized L2 curve and a 4800-5300 Angstrom
zoom on the same vertical scale. Labels identify its three largest local maxima, without smoothing
or physical line assignments. The original neuron order and all numerical weights are preserved.
All 48,711 deduplicated spectra enter sample-level diagnostics. The cached
raw-flux calculations remain independent of model changes.

With `--paper-dir`, generation copies figures, complete history, run configuration, bootstrap
reports, architecture audit, complexity and ablation results into the manuscript project.
The generator does not rewrite manuscript text.

## Preserved earlier artifacts

The original `models/paper_hyperparams_model.keras` has 527,699 parameters and heads
Teff=(48,32), [Fe/H]=(16), log g=(48), despite the old entry point declaring the winning templates.
The old constructor hardcoded different widths. It now uses the tuner's builder and honors all
declared head templates. A regression check confirmed that changing a declared head changes the graph.

Original models, predictions, bootstrap reports and the first independent curve rerun remain
preserved. They document the earlier experiment and are excluded from corrected article results.
The original correctly shaped `best_stellar_multitask_model.keras` is preserved as well.
