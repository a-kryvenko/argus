# Model revalidation — October 10, 2026

Training through 2024; selection January–September 2025; calibration October–December 2025.
Test: 2026 only; available observations end in September.
Six candidates evaluated. Production is unchanged.

## Errors at 24 hours

| Model | N | MAE | RMSE | 80% interval coverage |
| --- | ---: | ---: | ---: | ---: |
| DLinear speed, km/s | 5075 | 55.59 | 77.91 | 84.8% |
| ap | 2064 | 8.10 | 18.47 | 85.4% |
| Dst, nT | 6205 | 11.53 | 19.92 | 82.2% |
| F10.7, sfu | 258 | 9.22 | 13.02 | 81.0% |

| Event | N | Brier | Skill vs persistence |
| --- | ---: | ---: | ---: |
| Kp ≥4 | 2064 | 0.0971 | 41.4% |
| Kp ≥5 | 2064 | 0.0385 | 44.8% |
| Kp ≥6 | 2064 | 0.0150 | 42.7% |
| Bt, nT ≥5 | 5369 | 0.2197 | 41.7% |
| Bt, nT ≥10 | 5369 | 0.0753 | 49.7% |
| Bt, nT ≥15 | 5369 | 0.0192 | 47.0% |

## Remaining work

- PROSWIN-dependent models: audit training dates and retrain components.
- Southward IMF: prepare GONG inputs and replace training-set calibration.
- S10/M10/Y10: train forecasts and obtain 2026 data. GOES calibration was refitted but remains untested.
- JB2008/DTM2020: independent 2026 density observations; altitude loss remains unvalidated.

## Limitations

2026 was previously inspected: evaluation is retrospective. OMNI is revised; availability is assumed at interval end.
Metrics describe new candidates, not previous production models. F10.7 uses OMNI,
uniform averaging of overlapping models and residual calibration; DLinear reports calibrated q50.

[Methodology](../../METHODOLOGY.md) · [MLflow](http://127.0.0.1:5000/#/experiments/30).
Full metrics, coverage, confidence intervals and hashes: `data/revalidation/frozen/2026-10-10-revalidation-v1/`.
