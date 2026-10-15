# IMF revalidation — October 15, 2026

**Production unchanged: neither candidate passed the release criteria fixed before testing.**

Train 2011–2024; early stopping Jan–Sep 2025; Platt calibration Oct–Dec 2025. No base-model refit on 2025. Test: available 2026 observations, cutoff October 10 UTC; retrospective. Bt supports 1–24 h; Bs supports 1–48 h.

Brier at +24 h; lower is better. All available comparisons use identical rows.

| Target, nT | N | Candidate | Current production | Analog50 | Training climatology |
| --- | ---: | ---: | ---: | ---: | ---: |
| BT ≥5 | 5,369 | 0.21965 | 0.22057 | 0.23797 | 0.26864 |
| BT ≥10 | 5,369 | 0.07529 | 0.07599 | 0.07864 | 0.07936 |
| BT ≥15 | 5,369 | 0.01917 | 0.01937 | 0.01987 | 0.01944 |
| BS ≥5 | 5,369 | 0.04574 | — | 0.04745 | 0.04602 |
| BS ≥10 | 5,369 | 0.00575 | — | 0.00592 | 0.00574 |
| BS ≥15 | 5,369 | 0.00168 | — | 0.00172 | 0.00167 |

Bt: pooled primary Brier difference versus production -0.001015; 95% paired interval [-0.005398, 0.004161]. At P≥0.5, pooled Bt≥10 recall falls from 22.9% to 18.5%.

Bs: pooled 1–48 h improvement against climatology is unconfirmed. A complete 2026 replay of current Bs is unavailable because historical GONG inputs are missing.

Analog50 is a past-only Riley2017 adaptation, not an industry service. Targets are hourly-vector Bt and max(−hourly Bz,0), not southward duration. Revised OMNI and assumed interval-end availability do not establish native-L1 live accuracy.

Version 2 corrects serving float32 parity and incumbent Bt input precision; weights are unchanged. Version 1 and its inspected test remain retained. Calibration parity: 52,524 Bt / 104,472 Bs rows; test parity: 130,176 Bt / 258,560 Bs rows. Paired bootstrap: seven UTC days, 2,000 resamples.

Artifacts: `data/revalidation/frozen/2026-10-15-imf-v2/`; [methodology](../../METHODOLOGY.md).
