# Clio

Clio collects space-weather observations, fills historical gaps, normalizes numeric
measurements and stores solar images. Its HTTP API serves stored data to other services.

Data flows: [DATA_FLOWS.md](DATA_FLOWS.md).

## Observations

Schedules and source priorities: [project.yaml](../../configs/project.yaml).
Frequency means **collector polling**, not upstream publication cadence.
Live and historical sources are separate; Sources declared in priority order.

| Metric | Updates frequency | Source(s), ordered by priority | Notes |
| --- | --- | --- | --- |
| v | Live: 1 min; backfill: 6 h | Live: SWPC native RTSW plasma; history: OMNI, ACE, SOHO/CELIAS | Solar-wind speed, km/s; hourly history, 60-day backfill. |
| n | Live: 1 min; backfill: 6 h | Live: SWPC native RTSW plasma; history: OMNI, ACE, SOHO/CELIAS | Proton density, cm⁻³; hourly history, 60-day backfill. |
| t | Live: 1 min; backfill: 6 h | Live: SWPC native RTSW plasma; history: OMNI, ACE, SOHO/CELIAS | Proton temperature, K; hourly history, 60-day backfill. |
| b (`bt`) | Live: 1 min | SWPC RTSW magnetic feed; NOAA selects the active spacecraft | Total field, nT; native 1-minute L1 measurements, no propagation or configured historical fallback. |
| bz | Live: 1 min; backfill: 6 h | Live: SWPC native RTSW magnetic field; history: OMNI | GSM Bz, nT; hourly history, 60-day backfill. |
| kp | Live: 1 min; backfill: 6 h | Live: SWPC estimated Kp; history: OMNI | 3-hour index; 60-day backfill. |
| ap | Live: 1 min; backfill: 6 h | Live: SWPC Kp feed (`a_running`); history: OMNI | 3-hour ap, nT; 60-day backfill. |
| dst | Live: 5 min; backfill: 6 h | Live: Kyoto Dst via SWPC; history: OMNI | Hourly index, nT; 60-day backfill. |
| f10.7 (`f10_7`) | Live: 1 h; backfill: 1 day | Live: SWPC F10.7; history: GFZ, OMNI | Daily solar radio flux, sfu; 88-day backfill. |
| aia | Live: 5 min; warmup: 1 min | JSOC AIA synoptic NRT, shared SDO collector | Eight hourly 512×512 channels; 45-day retention. Native AIA193 FITS is retained alongside its reduced image for Prophet. |
| hmi | Live: 5 min | JSOC HMI `M_720s_nrt` FITS | Hourly 512×512 LOS magnetograms, Gauss; 45-day retention. |
| gong | Live: 1 h; backfill: 6 h | Live: `gong.live`; history: `gong.archive` | Original FITS magnetograms with file receipts; 2-day backfill. |

Native RTSW is the single live source for `v`, `n`, `t`, `bx`, `by`, `bz`, `bt`,
stored in `measurement` and used for normalization. `bx` and `by` follow the same configured policy as `bz`.
GOES source samples are archived as JSON snapshots: hourly live collection and
daily backfill over 88 days. Prophet derives `s10`, `m10`, `y10` from these samples.
Clio neither extracts AIA/GONG model features nor loads calibration artifacts.

Historical backfill fills gaps without overwriting existing observations.
SDO warmup fills retained hourly gaps in six-hour batches, with 60 seconds between
runs; cleanup runs hourly. SDO images retain source units and are not model-normalized.

## Commands

```bash
./argus clio worker                         # Run scheduled collection and processing
./argus clio collect                        # All configured observations
./argus clio collect v n t bz kp dst gong goes sdo
./argus clio backfill                       # Configured history windows
./argus clio backfill v n t --from 2026-08-04 --to 2026-08-05
./argus clio normalize
./argus clio status
./argus clio migrate
```
