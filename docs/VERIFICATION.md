# Local verification record

Software cleanup verified on 2026-10-16 using Windows and Python 3.11. Full
80-epoch retraining and another 100-trial search were not run or required.
These execution checks are not new scientific experiments.

| Check | Observed result |
| --- | --- |
| Pinned core plus optional notebook environment | Installation and pip check passed; Torch absent |
| Full regression/integration suite with Torch | 48 tests passed |
| Isolated source export without Torch, credentials, data or old models | 48 tests passed, including 3 explicit optional-Torch skips |
| Offline extended smoke | Main fit/evaluate/figures, OLS/Ridge/CNN/DNN, seven ablations, one tuning trial/finalfit passed |
| Cleared notebook | All 13 cells executed; tracked notebook remains cleared |
| Anonymous public sample | Downloaded real rows from train/validation/test; training, evaluation and figures passed |
| Frozen paper release | 14,443 test objects, 10,000 bootstrap resamples, ablation/bin reports and all paper figures regenerated |
| Keras checkpoint inference | Main, CNN and three DNN heads checked over all 14,443 test objects |
| Optional Transformer inference | Deterministic 64-object CPU float32 check passed against archived GPU predictions |
| Original research preservation | All 176 initially inventoried file hashes unchanged |
| Release bundle | 169 verified files; binary checkpoints/prediction archives unchanged |
| Published paper-v1 release | Anonymous empty-cache download, checksum verification and full reproduction passed without Torch |

Archived main MAE reproduced as 58.7176780438 K, 0.1013143523 dex and
0.1318624822 dex; individual bootstrap intervals match the archived values.
Keras forward comparisons use physical atol=0.002 and rtol=1e-6. The Transformer
check allows scaled atol=0.02 and rtol=0.005 for archived CUDA/bfloat16 versus CPU
float32; its saved-prediction metrics still use every test object. The archive's
known Transformer checkpoint-selection limitation is retained and documented.

The independent whole-branch review found three Important issues in non-default
paths: small-batch ablation LR scaling, sample-limit recovery metadata, and partial
ablation-report publication. Each was reproduced and fixed with a failing then
passing regression. Checkpoint/derived-output protection also covers linear-model
joblib files. Sample streaming failure has a regression preventing silent full-data
downloads. All checks above use separate output directories and preserve research data.

Windows and Ubuntu CI are configured. Linux was not run locally: WSL's VM service
reported HCS_E_SERVICE_NOT_AVAILABLE. Check GitHub Actions for the current remote
Windows and Ubuntu results.

The authorized public paper-v1 release is published at
https://huggingface.co/BrunoBarreto/sdss_dr12_stars_regression and pinned in
reference/artifacts.json to immutable commit
3a7b0b12020f577c49d806cd930979f3d5117c27. All 169 artifacts were downloaded into
an empty cache from an isolated source export with Hugging Face credentials removed.
Every file matched its size and SHA-256. The standard reproduce-paper command,
without --bundle, reproduced all 14,443 test objects, 10,000-bootstrap reports,
12 paper figures plus 2 auxiliary plots, and full Keras inference checks without Torch.
The existing verified public-data cache was reused; no historical research files
or model cache were supplied. Original checkpoints, predictions and scientific
results remain unchanged. Source publication proceeds through a dedicated GitHub
branch and pull request.
