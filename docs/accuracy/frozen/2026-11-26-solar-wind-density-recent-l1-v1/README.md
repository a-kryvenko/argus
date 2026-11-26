# Solar-wind density accuracy

Retrospective evaluation of `argus-plasma-density-recent-l1-v1` against native NOAA SWPC L1 measurements from 1 June to 25 November 2026. The replay assumes availability at hour close, requires ≥45 valid minutes per observed hour and uses up to 143 hours of history. Model selection and interval calibration used 2025 OMNI.

| Lead | Matched hours | Previous q50 MAE, cm⁻³ | Recent L1 q50 MAE, cm⁻³ |
| ---: | ---: | ---: | ---: |
| 1 h | 2,799 | 0.874 | 0.809 |
| 24 h | 2,765 | 2.323 | 2.199 |
| 96 h | 2,688 | 2.686 | 2.202 |

Among the 196 matched 24-hour forecasts with observed density ≥10 cm⁻³, q50 MAE rose from 9.99 to 11.20 cm⁻³.

[API metrics snapshot](metrics.json) · [Paired scores](paired.csv) · [Provenance](provenance.json)
