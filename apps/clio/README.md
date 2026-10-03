# Clio

Clio collects space-weather observations, fills historical gaps, normalizes numeric
measurements and stores solar images. Its HTTP API serves stored data to other services.

## Observations

Schedules and source priorities: [project.yaml](../../configs/project.yaml).
Frequency means **collector polling**, not upstream publication cadence.
Live and historical sources are separate; Sources declared in priority order.

| Metric | Updates frequency | Source(s), ordered by priority | Notes |
| --- | --- | --- | --- |
| v | Live: 1 min; backfill: 6 h | Live: SWPC propagated plasma; history: OMNI, ACE, SOHO/CELIAS | Solar-wind speed, km/s; hourly history, 60-day backfill. |
| n | Live: 1 min; backfill: 6 h | Live: SWPC propagated plasma; history: OMNI, ACE, SOHO/CELIAS | Proton density, cm⁻³; hourly history, 60-day backfill. |
| t | Live: 1 min; backfill: 6 h | Live: SWPC propagated plasma; history: OMNI, ACE, SOHO/CELIAS | Proton temperature, K; hourly history, 60-day backfill. |
| b (`bt`) | Live: 1 min | SWPC RTSW magnetic feed; NOAA selects the active spacecraft | Total field, nT; native 1-minute L1 measurements, no propagation or configured historical fallback. |
| bz | Live: 1 min; backfill: 6 h | Live: SWPC propagated magnetic field; history: OMNI | GSM Bz, nT; hourly history, 60-day backfill. |
| kp | Live: 1 min; backfill: 6 h | Live: SWPC estimated Kp; history: OMNI | 3-hour index; 60-day backfill. |
| ap | Live: 1 min; backfill: 6 h | Live: SWPC Kp feed (`a_running`); history: OMNI | 3-hour ap, nT; 60-day backfill. |
| dst | Live: 5 min; backfill: 6 h | Live: Kyoto Dst via SWPC; history: OMNI | Hourly index, nT; 60-day backfill. |
| f10.7 (`f10_7`) | Live: 1 h; backfill: 1 day | Live: SWPC F10.7; history: GFZ, OMNI | Daily solar radio flux, sfu; 88-day backfill. |
| aia | SDO: 5 min; separate `aia193`: live 1 h, backfill 6 h | SDO: JSOC AIA synoptic NRT; `aia193`: live `aia.nrt_193`, history `aia.synoptic_193` | SDO: hourly 512×512 images at 94, 131, 171, 193, 211, 304, 335, 1600 Å; 144-hour retention. Separate `aia193` FITS archive: 40-day backfill. |
| hmi | Live: 5 min | JSOC HMI `M_720s_nrt` FITS | Hourly 512×512 LOS magnetograms, Gauss; 144-hour retention. |
| gong | Live: 1 h; backfill: 6 h | Live: `gong.live`; history: `gong.archive` | Hourly magnetograms and stored features; 2-day backfill. |

Native RTSW also collects `v`, `n`, `t`, `bx`, `by`, `bz` every minute, separately
from propagated observations. `bx` and `by` follow the same configured policy as `bz`.
Additional solar indices `s10`, `m10`, `y10` use calibrated GOES data: hourly live
collection, daily backfill over 88 days.

Historical backfill fills gaps without overwriting existing observations.
SDO warmup fills retained hourly gaps in six-hour batches, with 60 seconds between
runs; cleanup runs hourly. SDO images retain source units and are not model-normalized.

## Commands

```bash
./argus clio worker                         # Run scheduled collection and processing
./argus clio collect                        # All configured observations
./argus clio collect v n t bz kp dst gong aia193
./argus clio backfill                       # Configured history windows
./argus clio backfill v n t --from 2026-08-04 --to 2026-08-05
./argus clio sdo-images live                # AIA/HMI hourly images
./argus clio sdo-images warmup              # Fill retained AIA/HMI gaps
./argus clio normalize
./argus clio aggregate
./argus clio status
./argus clio migrate
```
