# Fast L1 Kp/ap accuracy

January–June 2026 retrospective replay of `argus-geomagnetic-fast-l1-v1` and the latest available hourly forecast. Targets are complete three-hour GFZ intervals ending within six hours of issue. Wind is from archived SWPC L1 measurements with simulated receipt delays. At 15 minutes, 55 of 58 paired intervals used fast forecasts and three used hourly fallback. Ten intervals reached Kp ≥ 5.

| Metric, 15-minute delay | Hourly | Fast L1 with fallback |
| --- | ---: | ---: |
| Brier Kp ≥ 4 | 0.176 | 0.142 |
| Brier Kp ≥ 5 | 0.141 | 0.114 |
| Brier Kp ≥ 6 | 0.083 | 0.067 |
| ap median MAE | 20.47 | 17.03 |
| Kp ≥ 5 hits / false alarms | 4 / 0 | 4 / 1 |

Paired 95% intervals for fast minus hourly are [−0.074, +0.029] for Kp ≥ 5 Brier and [−5.52, −0.86] for ap MAE. The diagnostic Kp ≥ 5 cutoffs were selected in 2025 under a 5% false-positive limit, 13.64% for hourly and 36.12% for fast L1.

[API metrics](metrics.json) · [Scores at 5, 15 and 30 minutes](test_scores.csv) · [Paired intervals](paired_deltas.csv) · [Cutoff selection](selection_scores.csv) · [Replay manifest](replay_manifest.json) · [Provenance](provenance.json)

```sh
notebooks/.venv/bin/python -m notebooks.workflows.revalidation.cme_l1_kp_fast_model \
  --output /tmp/argus-fast-l1-accuracy
```
