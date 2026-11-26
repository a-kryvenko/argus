# Hourly Bt and Bs after L1 detections

January–June 2026 replay of `argus-bt-revalidated-2026-v2` and `argus-bs-revalidated-2026-v2` after 59 blind L1 detections. Hourly releases use inputs available at :10 UTC and are assumed available at :15. Scores cover full native L1 hours starting at or after the rapid issue and ending within six hours of it. Each observed hour averages at least 45 one-minute values of Bt magnitude or `max(−Bz, 0)` for Bs.

| Threshold at 15-minute input delay | Bt observed | Bt Brier | Bs observed | Bs Brier |
| --- | ---: | ---: | ---: | ---: |
| ≥ 5 nT | 225 | 0.1000 | 34 | 0.1139 |
| ≥ 10 nT | 99 | 0.1807 | 8 | 0.0300 |
| ≥ 15 nT | 34 | 0.0704 | 2 | 0.0076 |

The table covers 264 unique post-detection hours. Rolling verification scores scheduled forecasts across its 30-day window. The October revalidation used hourly-vector targets. [October revalidation](../2026-10-15-imf-v2/README.md) · [Fast L1 Bt comparison](../2026-11-26-bt-fast-l1-v1/README.md)

[Scores at 5, 15 and 30 minutes](scores.csv) · [Replay protocol](../2026-11-26-bt-fast-l1-v1/protocol.json) · [Provenance](provenance.json)
