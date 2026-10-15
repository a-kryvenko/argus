# Intelligence

API and worker for spacecraft density-risk advisories.
Public infrastructure only; calculations and risk policy live in private
`packages/intelligence-core`, accessed through `intelligence_core.api`.

## Dataflow

- **Input:** Density/CME forecasts from Prophet; trajectories, reference densities and requests from API.
- **Output:** Attention intervals and reasons, data statuses, assessment history and review journal over HTTP to API.

## Operation

Configure `INTELLIGENCE_DB_*`, `FORECASTS_URL`, `FORECASTS_SERVICE_TOKEN`
and `INTELLIGENCE_SERVICE_TOKEN` for the API.
Compose runs `intelligence-api` and `intelligence` (worker, 60-second polling).

```bash
./argus compose exec -T intelligence intelligence migrate upgrade head
./argus compose restart intelligence-api intelligence
```

Experimental loss monitoring is disabled by default:
`INTELLIGENCE_EXPERIMENTAL_DRAG_MONITOR=true` enables it.

[Advisory contract](../../docs/density-advisory.md) ·
[Experimental drag](../../docs/leo-drag.md) ·
[Deployment](../../README_DEPLOY.md) · [Commands](../../docs/commands.md)
