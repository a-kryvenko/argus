# Prophet

Publishes forecasts from saved Clio inputs. Cron commands calculate; HTTP serves stored releases.

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
| `geomagnetic-activity` | Kp probabilities and three-hour ap quantiles from one model, +1…48 h |
| `dst` | Dst quantiles |
| `hmf` | Total/southward IMF probabilities |
| `atmospheric-density` | Density grid; JB2008 model background |

Generation requires private `forecast-core` and configured models.
`cme_arrival.model` selects the shared arrival adapter for event and speed forecasts. `argus-cme-arrival-enlil-v1` passes through ENLIL arrival times; it is not a locally trained arrival model. Clio supplies external predictions; Prophet selects versions and forms forecast statuses.
`solar-radiation` has a read contract but no supported generator.
[Registry](../../configs/models_registry.yaml) · [Scheduler](src/prophet/scheduler.py).

Products publish independently; failures retain the previous release.
Cron runs events and summary every five minutes and regular forecasts hourly at :10.
The minute L1 detector can update Kp/ap for the next complete three-hour interval. Updated public points include `update_issued_at`. The release and its input snapshot are stored under `geomagnetic-fast-l1`.
Verification runs every six hours. The worker prepares rolling statistics at startup and every five minutes using `prophet.verification_days` (30 days).

Prophet prepares calibrated GOES indices and reads PROSWIN predictions before freezing the
input snapshot. Derivative cache: `data/prophet/source-cache`
(`PROPHET_FEATURE_CACHE` override); rebuildable from Clio's 45-day originals.

## HTTP

Go read API. Bearer token: `FORECASTS_SERVICE_TOKEN`.
Prefix `/internal/v1/forecasts/{product}`: `latest`, `releases/{id}`, `verification`.
All-product diagnostics: `/internal/v1/forecasts/status`.
`summary` saves `data/prophet/summary/latest.json` and uploads SVGs to public S3. `/internal/v1/summary` returns the prepared summary.
`/health/live` and `/health/ready` check process/storage, not forecast freshness.

[Setup](../../README_DEPLOY.md#local-development) · [Commands](../../docs/commands.md#prophet) · [Solar events](../../docs/solar-wind-events.md)

Forecasts and verification history are retained. Hourly `cleanup --apply` removes
input snapshots, provenance and errors after 7 days. Snapshot file entries contain
metadata only. `cleanup` previews the affected run count.
