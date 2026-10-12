# Validation methodology

[Protocol: argus-revalidation-2026-v1](../../scripts/revalidation/protocol.json). Applies to every registry model, JB2008 and DTM2020.

| Period | Use |
| --- | --- |
| 2011–2024 | Train models, preprocessing and climatology; record exceptions |
| January–September 2025 | Select architecture, features and hyperparameters; early stopping |
| October–December 2025 | Calibrate probabilities, intervals and ensemble weights |
| 2026 | Test frozen models only; no selection, calibration or refitting |

## Data and execution

- Use [required observations](../observation-requirements.md); retain source versions, SHA256, units, quality and observation/receipt times. Assumed availability makes an evaluation retrospective.
- Split by issue time, complete target interval and label availability; exclude boundary-crossing targets. Use only inputs available at issuance. Never fill missing targets; fix input-filling rules before testing.
- Verify pretrained components’ training cutoffs. Unknown cutoffs prevent a compliance claim. Do not refit base models on 2025.
- Audit every model, train, select, calibrate, verify inference parity, then freeze code, models, parameters and the test plan. Evaluate each frozen version once; retain prediction–observation pairs and blockers.
- For JB2008/DTM2020, prepare drivers and validate physical models; learned corrections follow the same splits. Density accuracy does not establish altitude-loss accuracy.

## Metrics

| Output | Required metrics |
| --- | --- |
| Point | MAE, RMSE, bias; skill against persistence and training climatology |
| Quantiles | q10/q50/q90 pinball; 80% coverage, width, tail misses and interval score |
| Probabilities | Brier and skill, ROC AUC, average precision, reliability; recall/FAR/CSI at P≥0.5 |
| Atmospheric density | MAE, RMSE, bias, MAPE, log-RMSE, underestimation >30%, upper-bound coverage |

Evaluate all horizons on identical model/baseline rows, without duplicating native-cadence targets. Report expected, observed, predicted and matched counts, unique targets, sampling cadence and fallbacks. Use paired seven-UTC-day block bootstrap (2,000 resamples; 95% intervals). AUC is undefined without both classes. Group density errors by Kp<3, 3≤Kp<5 and Kp≥5; future Kp is for scoring only.

## Results

- Data/models: `data/revalidation/<protocol>/<model>/`; immutable bundles: `data/revalidation/frozen/<run>/`. Corrections require a new version and reason.
- Public accuracy results: `docs/accuracy/`, Markdown only. Training logs, development audits and experiment reports stay under `data/revalidation/internal-reports/` or `data/mnt/revalidation/`, not public docs. Notebooks: `notebooks/revalidation/`; display saved results by default.
- [Local MLflow](http://127.0.0.1:5000): experiment `argus-revalidation-2026-v1`; save run IDs, parameters, split boundaries, hashes, per-horizon metrics and artifacts. Re-export must not rerun training or testing.

Current test cutoff: **2026-10-10 00:00 UTC**, limited by source coverage. Prior 2026 exposure makes this a retrospective test, not a blind holdout. Pending/blocked models are incomplete; production promotion is separate.
