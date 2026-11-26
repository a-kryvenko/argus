# Forecast accuracy

## Validation methodology

| Period | Use |
| --- | --- |
| 2011–2024 | Train models, preprocessing and climatology. Record exceptions. |
| January–September 2025 | Select architecture, features and hyperparameters, including early stopping. |
| October–December 2025 | Calibrate probabilities, intervals and ensemble weights. |
| 2026 | Evaluate frozen models. |

### Data and execution

- Retain observation versions, SHA-256 hashes, units, quality flags, observation times and receipt times. Label evaluations that assume receipt times as retrospective. [Required observations](../observation-requirements.md)
- Split by issue time, complete target interval and label availability. Exclude boundary-crossing intervals and missing targets. Use inputs available at issuance and fix input-filling rules before testing.
- Verify pretrained components’ training cutoffs before claiming compliance. Fit base models through 2024.
- Audit each model, then train, select, calibrate and check inference parity. Freeze code, data, model parameters and the test plan before scoring. Evaluate each frozen version once and retain prediction–observation pairs and blockers.
- For JB2008 and DTM2020, prepare drivers and validate the physical models. Apply the same splits to learned corrections. Evaluate atmospheric density and altitude loss separately.

### Metrics

| Output | Required metrics |
| --- | --- |
| Point | MAE, RMSE, bias and skill against persistence and training climatology |
| Quantiles | q10/q50/q90 pinball loss, 80% coverage, interval width, tail misses and interval score |
| Probabilities | Brier score and skill, ROC AUC, average precision, reliability and recall/FAR/CSI at P ≥ 0.5 |
| Atmospheric density | MAE, RMSE, bias, MAPE, log-RMSE, underestimation >30% and upper-bound coverage |

## Current models

- [Fast L1 Bt](frozen/2026-11-26-bt-fast-l1-v1/README.md) · [Metrics JSON](frozen/2026-11-26-bt-fast-l1-v1/metrics.json)
- [Hourly Bt/Bs after L1 detections](frozen/2026-11-26-hmf-hourly-v2/README.md)
- [Fast L1 Kp/ap](frozen/2026-11-26-kp-ap-fast-l1-v1/README.md) · [Metrics JSON](frozen/2026-11-26-kp-ap-fast-l1-v1/metrics.json)
- [48-hour Kp/ap](frozen/2026-11-26-kp-ap-v1/README.md)
- [Solar-wind density](frozen/2026-11-26-solar-wind-density-recent-l1-v1/README.md)
- [Solar-wind speed](frozen/2026-10-21-speed-release-v1/README.md)
- [CME correction in solar-wind speed](frozen/2026-10-15-cme-window-release-v1/README.md)
- [JB2008 / DTM2020 accuracy](ATMOSPHERIC.md)

## Earlier evaluations

- [October ap/Dst release](frozen/2026-10-15-ap-dst-release-v1/README.md)
- [October hourly Bt/Bs release](frozen/2026-10-15-imf-release-v1/README.md) · [Test results](frozen/2026-10-15-imf-v2/README.md) · [Bs with GONG](frozen/2026-10-15-imf-gong-v1/README.md)
- [October 12 plasma evaluation](frozen/2026-10-12-plasma-campaign-v2/README.md)
- [October 10 six-candidate evaluation](frozen/2026-10-10-revalidation-v1/README.md)
