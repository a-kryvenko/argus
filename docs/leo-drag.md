# Experimental LEO drag

Primary workflow: [density advisories on orbital parameters](density-advisory.md).
The API-only calculator and saved loss monitor are experimental;
flight-loss accuracy remains unvalidated.

## Calculator

`POST /api/v1/public/risks/leo-drag` (direct Uvicorn: omit `/api/v1`).

```json
{
  "altitude_km": 400,
  "inclination_deg": 51.6,
  "mass_kg": 100,
  "effective_area_m2": 1,
  "drag_coefficient": 2.2,
  "horizon_hours": 24
}
```

Supply actual spacecraft parameters. Optional `thresholds` compares cumulative
loss against user limits; without them `drag_risk` is `not_assessed`.
Response: density, drag acceleration/impulse, estimated altitude loss, hourly
predictions, echoed inputs, source and assumptions. `?meta=true` adds model metadata.

Limits: circular orbit, 200–800 km, 24/48 hours **from release issue time**,
constant area/Cd, fixed altitude and longitude-mean density. No maneuvers, winds or
orbit propagation. Estimates above 1,000 m are rejected as outside model scope.
Density spread and user-limit categories are not calibrated risk probabilities.
422: invalid/out-of-domain inputs; 503: unavailable, invalid or stale forecast/backend.

## Saved loss monitor

Session-owned routes under `/api/v1/dashboard/drag-monitor`; writes require Origin.
Maximum 100 profiles/account and five constant-area/Cd modes/profile.

| Method / suffix | Result |
| --- | --- |
| `GET /`, `POST /` | List/create profiles |
| `PUT /{id}?revision=N` | Replace profile; 409 on revision conflict |
| `POST /{id}/assess` | Calculate/reuse current assessment |
| `GET /{id}/history?limit=20` | Saved runs; maximum 50 |
| `GET /{id}/runs/{run_id}` | Exact immutable snapshot |
| `GET /{id}/chart?run_id=UUID` | Prepared chart; latest run by default |
| `GET /events?page=1&unacknowledged=false` | Journal; optional `profile_id` |
| `POST /events/{event_id}/ack` | Idempotent review acknowledgment |

The active mode determines attention. Missing/stale results and changed profiles
remain `not_assessed`. Crossings identify hourly samples, not precise event times.
Comparisons require matching profile inputs/revision and density model.
Runs and journal commit atomically; acknowledgment does not change risk status.
No external notifications or automatic history retention.

## Operation

Loss monitoring is off by default; enable worker polling with
`INTELLIGENCE_EXPERIMENTAL_DRAG_MONITOR=true`.
Scientific calculations, policy and validation tools belong to private
`intelligence-core`; public services handle transport and storage.

[Service setup](../apps/intelligence/README.md) · [Commands](commands.md) · [Density accuracy](accuracy/ATMOSPHERIC.md)
