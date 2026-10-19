# API

Public HTTP, dashboard authentication and account management.

## Dataflow

- **Input:** Client requests; Clio observations, Prophet forecasts, Intelligence advisories and static metrics.
- **Output:** HTTP data responses, dashboard sessions, profiles, history, events and monitoring.

## Endpoints

Paths below are relative to `/api/v1` at the proxy; direct Uvicorn omits that prefix.
Schemas and parameters: `/api/v1/docs`.

| Path | Purpose |
| --- | --- |
| `/observations` | Latest measurements with individual timestamps; `trends=true` adds observed changes |
| `/observations/history` | History, including normalized hourly indices; select variables with `observations=bx,bz,v,kp` |
| `/forecasts/{target}` | Published forecast |
| `/forecasts/{target}/metrics` | Static evaluation metrics |
| `/forecasts/{target}/verification` | Rolling observed accuracy |
| `/forecasts/atmospheric-density` | Density grid |
| `/forecasts/solar-wind-events` | [Solar events](../../docs/solar-wind-events.md) |
| `/dashboard/density-advisory/*` | [Owned profiles, assessments and journal](../../docs/density-advisory.md) |
| `/risks/leo-drag` | [Experimental calculator](../../docs/leo-drag.md) |
| `/dashboard/*` | Accounts, sessions, statistics and monitoring |
| `/ping` | Liveness |

Missing artifacts return 503.
Missing measurements remain null/gaps; history timestamps require a timezone.
Supported data routes accept `?meta=true` for units and descriptions.

[Setup](../../README_DEPLOY.md#local-development) · [Commands](../../docs/commands.md)
