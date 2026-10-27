# Solar event forecasts

UI: `/forecast/solar-events`. API paths below omit the proxy's `/api/v1` prefix.

| Product | Output |
| --- | --- |
| `/forecasts/solar-wind-events` | CME arrivals and L1 compression candidates |
| `/forecasts/solar-wind-speed` | Background quantiles plus conditional nominal/early/late CME speed scenarios |

HSS and DBEM forecasts are disabled. CME scenarios are not quantiles or calibrated
impact probabilities. Missing inputs mean unavailable context; compression alone
does not establish CME attribution. Unsupported multi-CME speed cases retain their
arrival forecasts and state the exclusion.

## Dataflow

Clio stores versioned DONKI/ENLIL receipts, hashes and normalized causal event inputs.
Prophet freezes inputs and publishes releases. Runtime algorithms and bundled CME
models live in private `forecast-core`; training/validation live in private notebooks.

Clio's `collect enlil` job archives compact NOAA NetCDF L1 profiles
from `noaa-wsa-enlil-pds.s3.amazonaws.com`, retaining creation, modification and local
receipt times. Discovery covers three days; stale/missing profiles remain explicit.
NOAA profiles are distinct from DONKI simulations and do not replace the speed model.
Units: proton-equivalent mass density in cm⁻³; velocity magnitude in km/s.

## Operation

Migrate Clio before updating Prophet (`20261014_enlil_profile` required).
Older snapshots without normalized event inputs are rejected; the speed background
remains available with CME context marked unavailable.

The approved CME model ships inside the private wheel/Prophet image; changing it
requires rebuilding Prophet. Optional `data/models/<model>.joblib` overrides must
match the registry SHA256. Missing inputs or a mismatched model disable the outlook.
Speed scenarios and baseline share one saved artifact; API never joins forecasts
from different releases.

## Validation limits

Train through 2024, select/calibrate in 2025, evaluate frozen models in 2026.
Historical 2026 results are retrospective, not blind holdouts. Archive modification
times are availability proxies, not original publication receipts. Numerical-profile
corrections fitted on 2025 are ineligible under this protocol; representative
pre-2025 compact profiles are still needed. Ambient/HSS peaks and overlapping windows
must not be reported as independent CME detection scores.
