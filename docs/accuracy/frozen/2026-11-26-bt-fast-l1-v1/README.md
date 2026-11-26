# Fast L1 Bt accuracy

January–June 2026 archive replay. At 15 minutes of simulated L1 delivery delay, fast Bt updated 264 paired one-hour targets after blind L1 detections. Targets are the mean of at least 45 minute Bt magnitudes. Fit through 2024, calibration in 2025 H1, selection in 2025 H2.

| Brier score at 15-minute delay | Observed | Hourly | Fast L1 |
| --- | ---: | ---: | ---: |
| Bt ≥ 5 nT | 225 | 0.100 | 0.070 |
| Bt ≥ 10 nT | 99 | 0.181 | 0.134 |
| Bt ≥ 15 nT | 34 | 0.070 | 0.059 |

For Bt ≥ 10 nT, the paired 95% interval for the Brier change is [−0.110, +0.027]. At diagnostic cutoffs selected in 2025, hits went from 53 to 51 and false alerts went from 11 to 14. The strict catalog-matched subset (±45 minutes, 44 hours) had Brier 0.474 → 0.198, with change [−0.548, −0.032].

[API metrics](metrics.json) · [Scores at 5, 15 and 30 minutes](scores.csv) · [Paired intervals](paired_deltas.csv) · [Hourly Bt/Bs](../2026-11-26-hmf-hourly-v2/README.md) · [Replay protocol](protocol.json) · [Provenance](provenance.json)
