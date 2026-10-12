# Plasma revalidation — October 12, 2026

**Speed, density and temperature candidate cycles completed. Upstream provenance remains partial.**
Train ≤2024; selection Jan–Sep 2025; calibration Oct–Dec 2025; locked retrospective test 2026.
Production model selection is unchanged.

## Test at +24 h

| Target | N | MAE | RMSE | 80% coverage | History-only MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Speed, km/s | 217 | 55.61 | 77.94 | 78.3% | 57.20 |
| Density, cm⁻³ | 217 | 2.30 | 4.75 | 82.0% | 2.69 |
| Temperature, K | 238 | 51982.15 | 79766.22 | 86.1% | 53887.60 |

History-only: speed/density DLinear; temperature T/n/v head. Comparisons use identical rows.
Sparse daily NRT targets, available through September 2026; N differs by input coverage.
Seven-day paired bootstrap intervals against persistence/climatology are saved; no significance claim against the history-only heads is made.

## Completed

- Temperature: 288 heads; both fallback branches separately scored.
- Density: retrained DLinear, 96 direct heads, recalibrated blend and intervals; DLinear fallback scored.
- Speed: reused revalidated DLinear; recalibrated blend, quantiles and probabilities for ≥450/500/600 km/s; fallback scored.
- Production-class parity passed for all horizons: speed/density 120 h, temperature 96 h. 42 tests passed.
- Source/code hashes, predictions, metrics, coverage and MLflow receipts frozen. Development cache inventory saved; no images downloaded.

Summary v2 corrects cache filename filtering only; model artifacts and scores are unchanged.

## Limits

One-hour assumed image latency forces history fallback at +96 h and beyond.
Archives are revised; historical receipt times are not established. Density/temperature train on author science forecasts and develop/test on cached NRT forecasts.
The published PROSWIN cutoff is June 2024. Reconstructed fold 01 matches 58,810 training samples, but scaler statistics differ (maximum mean difference 0.007788; scale 0.010713). Exact upstream provenance is unresolved, so full-chain compliance is not claimed.

Artifacts: `data/revalidation/frozen/2026-10-12-plasma-campaign-v2/`.
[Methodology](../../METHODOLOGY.md) · [Notebook](../../../../notebooks/revalidation/06_plasma_revalidation.ipynb).

## MLflow tests

- [2026-10-12-temperature-test-v1](http://127.0.0.1:5000/#/experiments/30/runs/b0b5e0bcd9794d8abc63a9adc0ced1af)
- [2026-10-12-density-test-v1](http://127.0.0.1:5000/#/experiments/30/runs/bc8340eb39ed4e4690c6a72330316b12)
- [2026-10-12-speed-blend-test-v1](http://127.0.0.1:5000/#/experiments/30/runs/3ab52a69d9424e09b75ed29184c4e071)
