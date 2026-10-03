# API

FastAPI serves Clio observations, Prophet forecasts, Intelligence drag assessments
and static model evaluation metrics. It also handles dashboard sessions and monitoring.

## Endpoints

Paths are relative to `/api/v1`. See `/api/v1/docs` for schemas, parameters and limits.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/ping` | Service liveness. |
| GET | `/public/observations/latest`, `/public/observations/history` | Normalized hourly observations. |
| GET | `/public/observations/solar-wind/latest`, `/public/observations/solar-wind/history` | Native L1 measurements and stored aggregates. |
| GET | `/public/observations/geomagnetic/latest`, `/public/observations/geomagnetic/history` | Three-hour Kp and hourly Dst. |
| GET | `/public/observations/summary` | Hourly changes and southward-Bz duration. |
| GET | `/public/observations/status` | Collection progress and freshness. |
| GET | `/{visibility}/forecasts/{target}` | Latest forecast. |
| GET | `/{visibility}/forecasts/{target}/metrics` | Model evaluation metrics. |
| GET | `/public/forecasts/atmospheric-density` | 48-hour JB2008 density grid. |
| POST | `/public/risks/leo-drag` | Circular-orbit drag assessment; [request and limits](../../docs/leo-drag.md). |

Forecast targets: `public` — `solar-wind-speed`, `solar-wind-density`,
`geomagnetic-activity`, `dst`; `private` — `hmf`, `solar-radiation`.
Atmospheric density uses its separate route. Unavailable forecast artifacts return 503.

Forecasts, metrics, density, drag, native observations and summaries support
`?meta=true` for descriptions and units in `data.meta`. Missing measurements remain
null or gaps. History timestamps must include a timezone.

Dashboard routes under `/dashboard` handle login/logout, the current user, user/group
management, usage statistics and [project monitoring](../../docs/project-monitoring.md).

## Operations

See [setup](../../README_DEPLOY.md#local-development), [commands](../../docs/commands.md)
and [observation sources](../clio/README.md#observations).

```bash
./argus api migrate
./argus api user --help
./argus logs api --tail 100
```
