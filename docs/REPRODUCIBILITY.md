# Reproducibility boundary and known limitations

The software starts from the pinned published Parquet benchmark. It does not
reconstruct the original survey/catalogue acquisition or establish independent
astrophysical ground truth. Public data, original archived arrays and manuscript
results remain unchanged by this cleanup.

## Data and labels

Inputs are published processed_flux vectors on the fixed 4,000-point logarithmic
wavelength grid. Targets use a training-only RobustScaler. Archived physical labels
are reconstructed exactly as float64(scaled_targets) * float64(scale) + float64(center).
Do not substitute unrounded catalogue columns when reproducing archived metrics.
Every comparison checks target order, ordered IDs, physical reference labels and,
when recorded, dataset fingerprints. Sample caches and synthetic fixtures are
marked separately and cannot impersonate the full reference dataset.

The original preprocessing policy retains some native-pixel issues. A prior audit
counted 438,325 affected native pixels in 11,859 of 48,711 spectra. A very close
cross-split sky pair was also flagged but not established as a duplicate. This
release neither silently changes masking nor revises the published split. Those
questions require separate scientific investigation.

## Archived versus future runs

The release preserves original checkpoints and prediction arrays byte for byte.
Portable JSON/CSV copies have separate hashes and record original source hashes.
The original training source commit was not recorded and remains unknown. New
runs record source file hashes, Git state when available, environment, seeds,
configuration, stage status and dataset identity. Deterministic settings do not
guarantee bitwise-identical GPU/CPU or cross-version optimization.

The original Transformer selected checkpoints using the mean of batch losses,
which underweights the last partial validation batch (750 objects). That archived
result is retained with this known limitation. New training uses object-weighted
validation loss, recorded as object_weighted_huber_v2. It is a corrected future
run, not a relabeling of the archived result. Optional CPU float32 checkpoint
checks allow documented tolerance against saved CUDA/bfloat16 predictions.

The paper uses a single training seed. Bootstrap comparisons quantify test-object
sampling uncertainty conditional on fixed checkpoints; they do not quantify
optimization/seed variation. Sparse parameter bins and SSPP label uncertainties
limit interpretation. Gaussian augmentation is a regularization perturbation,
not a full physical instrumental-noise simulator.

## Downloads and troubleshooting

Full downloads use ordinary HTTPS requests to the immutable Hugging Face resolve
URLs, with three bounded attempts, SHA-256 verification and atomic completion.
This is also the HTTP fallback when Xet-specific tools fail. Source identities and
all three known checksums are in reference/dataset.json. A corrupted existing file
is preserved and rejected: choose a fresh --data-dir or move it aside explicitly.
For network-free use, retain the verified Parquet files, compact NPZ and provenance
manifest together, then pass --offline. Sample rows are streamed anonymously from
the pinned revision and stored under data/samples, never over full Parquet files.
Sample streaming also has three bounded attempts. Failure does not silently switch
to a multi-GB transfer; request prepare-data explicitly to use the full HTTP route.

Training output collisions are errors. Append only unrun baseline/ablation stages.
Derived replacement is explicit and cannot target directories containing model
checkpoints or histories. For further figures use --output-dir NEW_DIRECTORY.

Windows and Ubuntu CI run offline tests and miniature training. Substantial public
transfers and complete reference reproduction use a separately requested workflow.
No automatic publication, full retraining or paid compute is part of ordinary tests.

Local cleanup verification used Windows Python 3.11. Ubuntu CI is configured;
local WSL execution was unavailable because its VM service could not start.
