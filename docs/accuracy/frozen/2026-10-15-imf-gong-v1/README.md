# Bs accuracy with historical GONG

The [protocol-trained candidate](../2026-10-15-imf-v2/README.md) is compared with
current `argus-bs-t-v2` on 252,048 identical issue/horizon pairs in 2026, through
September 3. Another 6,512 pairs lack usable GONG and are excluded from all models.
Candidate probabilities are reused; no refitting or candidate rescoring.

GONG: 5,721 maps, December 30, 2025–September 2, 2026. Exact observation times,
assumed one-hour availability delay, maximum observation age three hours.
Historical publication receipts remain unknown; this is a retrospective replay.
Current Bs training provenance remains unestablished.

Brier at +24 h; lower is better. N = 5,233 per model.

| Bs threshold, nT | Candidate | Current production | Analog50 | Training climatology |
| --- | ---: | ---: | ---: | ---: |
| ≥5 | 0.04520 | 0.04713 | 0.04700 | 0.04559 |
| ≥10 | 0.00589 | 0.00592 | 0.00607 | 0.00589 |
| ≥15 | 0.00172 | 0.00176 | 0.00176 | 0.00172 |

**Production unchanged.** Pooled 1–48 h primary Brier difference versus production:
−0.001357; paired 95% interval [−0.003441, +0.000392]. Improvement is unconfirmed;
pooled ≥10 nT Brier worsens, with supported regressions in 13 horizon/threshold pairs.
Seven-UTC-day paired bootstrap: 2,000 resamples. Legacy serving parity: 1,025 rows.

FITS: `data/mnt/raw/gong`; download metadata/features:
`data/mnt/revalidation/gong-2026-v1`. Frozen comparison:
`data/revalidation/frozen/2026-10-15-imf-gong-v1/`.
