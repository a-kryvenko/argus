# Forecast accuracy

[Latest snapshot](snapshots/2026-10-07.md) · [Settings](../../scripts/benchmarks/protocol.json)

## Evaluation periods

| Settings name | Training years, inclusive | Evaluation year |
|---|---|---|
| `train2024_test2025` | 2011–2024 | 2025 |
| `train2025_test2026` | 2011–2025 | 2026 |
| `train2024_test2026_control` | 2011–2024 | 2026 |

Compare the second and third rows on identical 2026 dates with identical model
settings to measure the effect of adding 2025 training data. Evaluation dates
are excluded from fitting, parameter selection and calibration within each row.
2025 and parts of 2026 have been used in earlier research. The current observation
cutoff is October 7, 2026, 00:00 UTC; 2026 is incomplete.

## Data and metrics

- OMNI: hourly V/N/T, Dst and IMF; three-hour Kp/Ap.
- Bt: magnitude of the hourly mean vector. Bs: `max(0, -Bz_GSM)` using hourly Bz.
- SOLFSMY: daily F10.7/S10/M10/Y10, without hourly duplication.
- Missing targets stay missing. Target intervals must remain inside their split.
- Assumed availability: interval end for OMNI; day end +24h for SOLFSMY.
  Historical publication timestamps are unavailable.

Every evaluation requires persistence and climatology on the same examples as
the service forecast. Their results and model details are stored in MLflow.

| Reference forecast | Calculation |
|---|---|
| Persistence | Latest available value; maximum age 24h for OMNI, 72h for SOLFSMY. |
| Climatology | Training mean for point forecasts, empirical quantiles for intervals, event frequencies for probabilities. |

Point metrics: MAE, RMSE, bias and sample count. Probability metrics: Brier and
ROC AUC per threshold. Quantile metrics: pinball, interval coverage and width.
Relative improvement: `1 - service_error / reference_error`.
Persistence uses identical quantiles and threshold probabilities of zero or one.
Undefined metrics and missing evaluations are shown as `—`, never zero.

## Snapshots

`docs/accuracy/snapshots/YYYY-MM-DD.md` records service accuracy as of that date.
Each snapshot contains tables for 3, 6, 12, 24, 48 and 96 hours, with separate
2025 and 2026 columns. N is the number of evaluated forecast–observation pairs.
Snapshots contain service goals and metrics; model identifiers, source files and
training details belong in MLflow. CSV files are local and are not linked from snapshots.

The October 7 snapshot uses recorded 2025 evaluations for speed and density,
and the notebook evaluator on prepared 2025 data for Ap, Dst, F10.7, Kp and Bs.
These service evaluations use their existing data and training periods, not the
new 2011-based training experiment. The available Bt scores combine multiple
years and cannot be assigned to the 2025 column. No 2026 service evaluation is
available. Missing goals remain listed with empty results.

The separate reference calculations use native-cadence observations and different
samples. They cannot establish improvement over persistence or climatology until
service predictions are evaluated on the same pairs. SOLFSMY 2026 and verified
flare, DTC and atmospheric-density targets are not available in that calculation.

## Commands

Run from the repository root. Output directories must be new.

```bash
.venv/bin/python -m scripts.benchmarks.benchmark \
  --output data/metrics/benchmarks/<run-id>

.venv/bin/python -m scripts.benchmarks.compare \
  --benchmark data/metrics/benchmarks/20261007-v1 \
  --fold train2024_test2025 \
  --predictions /path/to/predictions.parquet \
  --metadata /path/to/model.json \
  --output data/metrics/benchmarks/<model-run-id>
```

The first command calculates persistence and climatology from local observations.
The second compares saved point forecasts. Prediction fields: `target`,
`issue_time`, `valid_time`, `lead_hours`, `prediction`. Timestamps require a timezone.
Metadata fields: `model_sha256`, `train_start`, `fit_end`, `selection_end`,
`calibration_end`, `upstream_training`, `prior_test_exposure`,
`target_definition: "forecast-benchmark-v1"`. End dates are exclusive; training
through 2024 ends at `2025-01-01T00:00:00Z`. Without calibration, use `fit_end`.

## Storage and MLflow

Local results: `data/metrics/benchmarks/<run-id>/`. Keep metrics, observation pairs,
coverage, settings and file hashes. Existing results are not overwritten.
`data/` is excluded from Git. Service snapshot calculations also use
`data/metrics/service-snapshots/`. Earlier per-product results remain in place.

```bash
.venv/bin/python -m scripts.benchmarks.mlflow_export \
  --report data/metrics/benchmarks/20261007-v1 \
  --experiment forecast-benchmarks
```

Tracking address: `--tracking-uri`, then `MLFLOW_TRACKING_URI`, otherwise
`http://localhost:5000`. One run stores the report; child runs separate evaluation
period, target and forecast method. Metric `step` is the lead in hours.
`--include-pairs` also uploads observation and prediction parquet files.
Each export creates a new run without changing local results.

Tests: `.venv/bin/python -m pytest scripts/benchmarks/test_*.py -q`.
