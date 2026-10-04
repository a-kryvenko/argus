# Prophet

Data flows: [DATA_FLOWS.md](DATA_FLOWS.md).

Prophet generates forecasts from Clio observations, saves input snapshots and results
in PostgreSQL, and publishes releases over HTTP. The `prophet` worker runs scheduled
calculations; `prophet-api` serves stored releases.

## Products

| Product | Output |
| --- | --- |
| `solar-wind-speed` | Speed quantiles and threshold probabilities. |
| `solar-wind-density` | Proton density quantiles. |
| `geomagnetic-activity` | Kp threshold probabilities and Ap quantiles. |
| `dst` | Dst quantiles. |
| `hmf` | Total and southward magnetic-field threshold probabilities. |
| `atmospheric-density` | JB2008 atmospheric density grid. |

Generation requires configured models and the private forecast-core backend.
Model settings are in [models_registry.yaml](../../configs/models_registry.yaml);
schedules and input policies are in [project.yaml](../../configs/project.yaml).
`solar-radiation` has a read contract but no supported generator.

Products run hourly at `:10 UTC` by default, with up to two calculations in parallel.
Products dispatched together share one Clio snapshot. Each complete product publishes
independently; failures retain its previous release. Manual runs do not consume scheduled slots.

Published-release verification runs every six hours over the preceding seven days.
It compares forecasts with Clio observations and stores results separately from static
website model metrics. Atmospheric density is excluded from verification.

## HTTP

Forecast routes require a Bearer token matching `FORECASTS_SERVICE_TOKEN`.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/internal/v1/forecasts/{product}/latest` | Current complete release. |
| GET | `/internal/v1/forecasts/{product}/releases/{release_id}` | Historical release. |
| GET | `/internal/v1/forecasts/{product}/verification` | Rolling 30-day observed accuracy, grouped by artifact and model hash. |
| GET | `/internal/v1/forecasts/{product}/status` | Release age, latest attempt and input diagnostics. |
| GET | `/health/live`, `/health/ready` | Service liveness and storage readiness. |

## Commands

```bash
./argus prophet generate                    # All supported products; refresh is an alias
./argus prophet generate solar-wind-speed
./argus prophet worker
./argus prophet status                      # All products
./argus prophet verify all --days 7
./argus prophet verification-report all
./argus prophet cleanup --days 90           # Preview; --apply deletes eligible history
./argus prophet migrate
```

See [setup](../../README_DEPLOY.md#local-development), [commands](../../docs/commands.md)
and [forecast workflows](../../docs/forecast-workflows.md).

## Observation preparation

Prophet reads Clio's source-file catalog, downloads checksum-verified originals,
and prepares AIA/GONG features and calibrated GOES solar indices before storing
its forecast input snapshot. Clio supplies no model features or calibration code.
The active southward IMF model uses GONG latitude-band features; its three-hour
freshness policy is unchanged. Source receipt times remain the causality boundary.
AIA masks and downloaded originals are cached under `data/prophet/source-cache`
(or `PROPHET_FEATURE_CACHE`). Calibration artifacts stay in `models_registry.yaml`.

AIA originals now come exclusively from Clio's shared SDO archive (45 days).
Prophet selects a causal 40-day source history for temporal comparisons. Its cache
is optional and rebuildable from Clio, including rotation pairs after a cold start.
Cache files older than 45 days are removed after input preparation. Missing source
observations retain the model's existing DLinear fallback; source availability
times remain unchanged.
