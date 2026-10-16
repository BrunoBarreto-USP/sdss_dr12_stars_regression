# Single-column figures and legacy command adapters

Use `python -m stellar_workflow` from the repository root. Training, reports and
figures share one run directory; new training never needs an older paper model.

| Legacy entry point | Shared command |
| --- | --- |
| scripts/retrain_paper_with_history.py --out-dir RUN | train --run-dir RUN |
| scripts/finalize_corrected_experiment.py --run-dir RUN | evaluate --run-dir RUN |
| scripts/retrain_corrected_ablations.py --run-dir RUN | ablations --run-dir RUN |
| scripts/generate_paper_figures.py --run-dir RUN | figures --run-dir RUN |
| scripts/analyze_parameter_ranges.py --run-dir RUN | figures --run-dir RUN |
| scripts/audit_research_pipeline.py --run-dir RUN | audit --run-dir RUN |

The old figure `--out-dir` option aliases `--output-dir`. `--only-training` writes
a dedicated training-figures directory by default. Parameter-bin plots are included
in the complete set of figures. `--summarize-only` on the ablation adapter reads all
seven saved variants and writes the same paired summary as reference reproduction.

The generator preserves STIX fonts, horizontal dimensions, scatter transparency,
SNR color normalization, weight-matrix contrast/zoom and architecture diagrams.
Ordinary runs support sample data and tuned widths. `--paper-reference` alone
enforces the frozen full-data counts and architecture.

```powershell
python -m stellar_workflow figures --run-dir outputs/runs/example
python -m stellar_workflow figures --run-dir outputs/runs/example --output-dir outputs/figures/revision2
python -m stellar_workflow figures --run-dir outputs/runs/example --replace-derived
```

Paper-source export is opt-in using `--paper-dir PATH`. It copies generated figures
and reproducibility reports; omit this option when checking the repository alone.
Full installation, data preparation and archived reproduction are documented in the
[main README](../README.md).
