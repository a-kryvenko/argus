# Prophet

Publishes forecasts from saved Clio inputs. Worker calculates; HTTP serves stored releases.

## Dataflow

- **Input:** Observations, original solar files and event catalogs from Clio; configured model artifacts.
- **Output:** Versioned forecasts, density/reference grids, verification results and publication status over HTTP to API and Intelligence.

[DATA_FLOWS.md](DATA_FLOWS.md).

## Products

| Product | Output |
| --- | --- |
| `solar-wind-events` | CME arrivals, conditional speed scenarios and L1 event context |
| `solar-wind-speed` | Speed quantiles and threshold probabilities |
| `solar-wind-density` | Density quantiles |
| `solar-wind-temperature` | Temperature quantiles |
| `geomagnetic-activity` | Kp probabilities and Ap quantiles |
| `dst` | Dst quantiles |
| `hmf` | Total/southward IMF probabilities |
| `atmospheric-density` | Density grid; JB2008 model background |

Generation requires private `forecast-core` and configured models.
`cme_arrival.model` selects the shared arrival adapter for event and speed forecasts. `argus-cme-arrival-enlil-v1` passes through ENLIL arrival times; it is not a locally trained arrival model. Clio supplies external predictions; Prophet selects versions and forms forecast statuses.
`solar-radiation` has a read contract but no supported generator.
[Registry](../../configs/models_registry.yaml) · [Schedules](../../configs/project.yaml).

Products publish independently; failures retain the previous release.
Default schedule: hourly at :10 UTC, two concurrent calculations.
Verification runs every six hours over seven days; atmospheric density is excluded.

Prophet prepares AIA/GONG features and calibrated GOES indices, then freezes the
input snapshot. Derivative cache: `data/prophet/source-cache`
(`PROPHET_FEATURE_CACHE` override); rebuildable from Clio's 45-day originals.

## HTTP

Bearer token: `FORECASTS_SERVICE_TOKEN`.
Prefix `/internal/v1/forecasts/{product}`: `latest`, `releases/{id}`, `status`, `verification`.
`/health/live` and `/health/ready` check process/storage, not forecast freshness.

[Setup](../../README_DEPLOY.md#local-development) · [Commands](../../docs/commands.md#prophet) · [Solar events](../../docs/solar-wind-events.md)
