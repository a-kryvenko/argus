# Forecast accuracy report

Snapshot: **2026-09-29**. All results below are for a **24-hour lead**, not an
average over the forecast horizon. These are saved historical evaluations;
no models were retrained or reevaluated for this report. Live verification follows
a separate [protocol](forecast-workflows.md).

MAE/RMSE measure q50 error in the target's units; lower is better. Coverage is the
observed fraction inside q10–q90 (nominally 80%), not a percentage of prediction
accuracy. Brier measures probability error (lower is better); ROC AUC measures
ranking (higher is better). N counts forecast–observation pairs, not independent
events. Missing evaluation evidence is stated explicitly.

## Solar wind speed — v (km/s)

- **Model:** `argus-plasma-speed-aia-ridge-v1`.
- **Training data:** frozen DLinear trained on 1995–1999 solar wind data;
  Ridge correction trained on 2023–2024 OMNI targets and AIA193 image features.
  Uncertainty uses temporal validation residuals from Q2–Q4 2024; those quarters
  also informed correction-scale selection, so calibration is not independent.
- **Test data:** observed OMNI targets in 2025, releases every 6 hours, common
  sample with available AIA. N = 1,447 at 24 h (172,820 pairs across 1–120 h).
- **Results:** MAE **58.188 km/s**, RMSE **81.903 km/s**, coverage **64.1%**.

| Event | Brier | ROC AUC |
|---|---:|---:|
| v ≥ 450 km/s | 0.1573 | 0.852 |
| v ≥ 500 km/s | 0.1568 | 0.836 |
| v ≥ 600 km/s | 0.0989 | 0.819 |

2025 was repeatedly explored; this is not an independent test. These results do
not evaluate the no-AIA fallback. Sources: [training protocol](../scripts/training/aia_wind/README.md),
[metrics](../data/metrics/plasma/README.md), [artifact hash](../data/metrics/plasma/frozen.json),
[evaluation settings](../data/metrics/plasma/evaluation.json).

## Solar wind proton density — n (cm⁻³)

- **Model:** `argus-plasma-den-q-v2`.
- **Training data:** notebook specifies `data/training/2010_2024`, with n, v,
  Bt and F10.7 plus 3 h, 6 h and 7-day summaries. Chronological 85%/15% split
  with a 120 h purge; the calibration partition is not used. The original
  training snapshot has not been recovered and verified against the artifact.
- **Test data:** existing `data/training/2025` features, hourly releases;
  rows without observed target_n or finite current n excluded. N = 8,656 at 24 h.
- **Results:** MAE **2.797 cm⁻³**, RMSE **5.266 cm⁻³**, coverage **74.7%**.
  Persistence MAE: 3.730; historical-median baseline MAE: 3.166 cm⁻³.

Historical diagnostic, not an independent holdout or live-service evaluation.
The older density metrics CSV does not reproduce and is excluded. Sources:
[training notebook](../notebooks/4_train_density_quantiles.ipynb),
[audit](solar-wind-density-audit-2026-09-29.md),
[results](../data/reports/density_audit_20260929/selected_leads.csv),
[artifact hash and protocol](../data/reports/density_audit_20260929/provenance.json).

## Geomagnetic Ap index

- **Model:** `argus-ap-q-v2`.
- **Training data:** notebook specifies `data/training/2010_2024`; Ap, Dst,
  southward Bz, Bt, solar wind speed summaries and F10.7. Chronological
  85%/15% training/calibration split, no purge configured.
- **Test data:** shared evaluator defaults to `data/training/2025`, excluding
  missing current/target values. The saved CSV does not establish the actual
  dataset snapshot, evaluation period, N or artifact hash.
- **Saved results:** MAE **9.434**, RMSE **18.533**, coverage **68.3%**.

Training recipe and evaluation defaults are not verified provenance for these
numbers. Sources: [notebook](../notebooks/4_train_ap_quantiles.ipynb),
[metrics](../data/metrics/geomagnetic/ap_quantile/regression.csv).

## Geomagnetic Dst index (nT)

- **Model:** `argus-dst-q-v2`.
- **Training data:** notebook specifies `data/training/2010_2024`; Dst, Ap,
  southward Bz, Bt, solar wind speed summaries and F10.7. Chronological
  85%/15% training/calibration split, no purge configured.
- **Test data:** shared evaluator defaults to `data/training/2025`, excluding
  missing current/target values. Actual dataset snapshot, period, N and artifact
  hash are not recorded in the saved CSV.
- **Saved results:** MAE **12.822 nT**, RMSE **18.384 nT**, coverage **53.1%**.

Provenance remains unverified. Sources: [notebook](../notebooks/4_train_dst_quantiles.ipynb),
[metrics](../data/metrics/geomagnetic/dst_quantile/regression.csv).

## Solar radio flux — F10.7 (sfu)

- **Model:** `argus-f10_7-q-v1`.
- **Training data:** notebook specifies `data/training/2010_2024`; F10.7 history,
  solar wind v/n and magnetic-field features. Chronological 85%/15%
  training/calibration split with a 120 h purge.
- **Test data:** shared evaluator defaults to `data/training/2025`, excluding
  missing current/target values. Actual dataset snapshot, period, N and artifact
  hash are not recorded in the saved CSV.
- **Saved results:** MAE **8.416 sfu**, RMSE **13.772 sfu**, coverage **68.7%**.

Provenance remains unverified. Sources: [notebook](../notebooks/4_train_f10_7_quantiles.ipynb),
[metrics](../data/metrics/radiation/f10_7_quantile/regression.csv).

## Geomagnetic Kp index

- **Model:** `argus-kp-t-v2`.
- **Training data:** notebook specifies `data/training/2010_2024`; Kp, solar wind
  speed, Dst, F10.7 and temporal summaries. Chronological 85%/15%
  training/calibration split, no purge configured.
- **Test data:** shared evaluator defaults to `data/training/2025`, excluding
  missing current Kp or threshold targets. Actual dataset snapshot, period, N
  and artifact hash are not recorded in the saved CSVs.

| Event | Brier | ROC AUC |
|---|---:|---:|
| Kp ≥ 4 | 0.1702 | 0.525 |
| Kp ≥ 5 | 0.0526 | 0.460 |
| Kp ≥ 6 | 0.0134 | 0.499 |

Provenance remains unverified. Sources: [notebook](../notebooks/9_train_kp.ipynb),
[metrics](../data/metrics/geomagnetic/kp_proba/).

## Total interplanetary magnetic field — Bt (nT)

- **Model:** `argus-bt-t-v3`.
- **Training data:** final model trained on 2020–2024 OMNI data, with Platt
  calibration on 2025. Inputs include hourly magnetic-field and solar wind
  history; the rolling-history archive spans 2013–2025.
- **Test data:** metrics are labelled as a 2020–2025 walk-forward evaluation
  with annual training/calibration folds. CSV N = 8,683 at 24 h. This evaluates
  the training recipe, not an independent test of the final artifact calibrated
  on 2025; the provenance file does not enumerate the fold boundaries.

| Event | Brier | ROC AUC |
|---|---:|---:|
| Bt ≥ 5 nT | 0.2095 | 0.743 |
| Bt ≥ 10 nT | 0.0620 | 0.753 |
| Bt ≥ 15 nT | 0.0153 | 0.721 |

Sources: [release report](../notebooks/research/imf_total_v3_release.md),
[metrics](../data/metrics/hmf/total_proba_v3/),
[provenance](../data/metrics/hmf/total_proba_v3/provenance.json).

## Southward interplanetary magnetic field — Bs = max(−Bz, 0) (nT)

- **Model:** `argus-bs-t-v2`.
- **Training data:** notebook specifies `data/training/2010_2024`; magnetic-field,
  plasma, Kp/Dst/F10.7 history and spatial magnetic-map features. Chronological
  85%/15% training/calibration split, no purge configured.
- **Test data:** shared evaluator defaults to `data/training/2025`, excluding
  missing current/threshold targets. Actual dataset snapshot, period, N and
  artifact hash are not recorded in the saved CSVs.

| Event | Brier | ROC AUC |
|---|---:|---:|
| Bs ≥ 5 nT | 0.0642 | 0.574 |
| Bs ≥ 10 nT | 0.0077 | 0.492 |
| Bs ≥ 15 nT | 0.0015 | 0.573 |

Provenance remains unverified. Sources: [notebook](../notebooks/8_train_southward_bz.ipynb),
[metrics](../data/metrics/hmf/southward_proba/).

## Solar S10 index

- **Model:** `argus-s10-q-v1`.
- **Training data:** forecast training provenance is not documented in this snapshot.
- **Test data:** no saved forecast evaluation found.
- **Results:** unavailable. GOES → SOLFSMY observation-calibration metrics do not
  measure forecast accuracy.

## Solar M10 index

- **Model:** `argus-m10-q-v1`.
- **Training data:** forecast training provenance is not documented in this snapshot.
- **Test data:** no saved forecast evaluation found.
- **Results:** unavailable. GOES → SOLFSMY observation-calibration metrics do not
  measure forecast accuracy.

## Solar Y10 index

- **Model:** `argus-y10-q-v1`.
- **Training data:** forecast training provenance is not documented in this snapshot.
- **Test data:** no saved forecast evaluation found.
- **Results:** unavailable. GOES → SOLFSMY observation-calibration metrics do not
  measure forecast accuracy.

## Temperature correction — DTC

- **Model:** active version is not pinned in the registry.
- **Training data:** not documented in this snapshot.
- **Test data:** no saved evaluation found.
- **Results:** unavailable.

## Atmospheric density — ρ (kg/m³)

- **Method:** JB2008 with forecast drivers.
- **Training data:** empirical physical model; a project-specific fitting dataset
  is not documented in this snapshot.
- **Test data:** no independent density observations matched to forecast
  coordinates are available in the current verification contract.
- **Results:** independent forecast accuracy has not been established.

## Maintaining this report

Model names follow the [registry](../configs/models_registry.yaml). Legacy
notebook split/default-evaluation descriptions follow the linked notebooks and
[shared evaluator](../notebooks/shared.py); they do not prove which data produced
an existing artifact or CSV. A low Brier score for a rare event alone does not
establish skill over an event-frequency baseline. Results from different samples
are not directly comparable.

For each new evaluation, append a dated snapshot and preserve earlier results.
Keep a separate block per variable with model/hash, training and calibration
data, test period and dataset version, protocol, lead, N, metrics and source
links. Record missing evidence explicitly; never replace missing metrics with zeros.
