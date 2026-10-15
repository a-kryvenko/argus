# Historical demo

Route: `/demo`. Hourly replay T−96…T0, where T0 is the
[G4 onset, January 19, 2026, 19:38 UTC](https://swpc-drupal.woc.noaa.gov/news/g4-severe-geomagnetic-storm-levels-reached-19-jan-2026).

## Build or update

Requires Docker Compose, installed Prophet models and scenario-pinned caches.

```bash
./argus demo                  # Cached archives
./argus demo --refresh-data   # Redownload and rebuild
./argus demo january-2026     # Named scenario
```

Scenarios: `configs/demo/{name}.json`; update hashes/cutoff evidence after model changes.
Build publishes 97 issues/product atomically to `data/demo/current.json`; failures
preserve the previous bundle. Release deployment uploads the ready bundle, not
archives. Reload `/demo` after updating. No automatic rebuild or operational scoring.

| Product | Output | Horizon |
| --- | --- | --- |
| Speed | Quantiles and probabilities | 96 h |
| Density | Quantiles | 96 h |
| Kp/Ap | Probabilities/quantiles | 48 h |
| Dst | Quantiles | 48 h |
| Bt | Probabilities | 24 h |
| F10.7 | Quantiles | 48 h |

## Limits

Revised OMNI2 archives do not reproduce real-time delays. Only completed intervals
feed forecasts; training/calibration must end before T−96. Legacy Kp/Ap/Dst/F10.7
cutoffs depend on recipes/MLflow, not embedded dates.

Speed/density need the pinned `data/demo/january-2026/proswin-science-hourly.json`
from author-preprocessed images, with per-lead fallback. Bs lacks historical GONG;
S10/M10/Y10 lack installed models.
