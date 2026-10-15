# Atmospheric density advisories

`/dashboard/risk/leo` shows density relative to a reference along supplied
ephemerides, with separate CME arrival context. No orbit propagation or altitude-loss estimate.

## Inputs

Upload trajectory JSON; the editor sets name, thresholds, maximum gap and enabled state.
2–5,000 strictly increasing UTC-aware samples; span ≤48 hours.

```json
{
  "trajectory": {
    "source": "Ephemeris provider / version",
    "generated_at": "2026-10-15T10:00:00Z",
    "valid_until": "2026-10-15T15:00:00Z",
    "coordinates": "geocentric_spherical",
    "points": [
      {"time": "2026-10-15T12:00:00Z", "altitude_km": 400, "latitude_deg": 0, "longitude_deg": 10},
      {"time": "2026-10-15T12:10:00Z", "altitude_km": 401, "latitude_deg": 20, "longitude_deg": 30}
    ]
  }
}
```

Format example only: replace times/positions with real inputs. Altitude is km;
latitude/longitude are geocentric degrees. Convert geodetic inputs before upload.
Generation/preparation times cannot be future; validity must cover the last sample.
Edits increment profile revision; expired trajectories require new ephemerides.

| `reference_mode` | Reference |
| --- | --- |
| `auto` (default) | Supplied operator reference, otherwise Prophet background |
| `prophet_background` | Published JB2008 model background; no manual fallback |
| `operator_reference` | Supplied per-point `reference_density_kg_m3` |

Manual references also require `reference` metadata: `label`, `source`,
`prepared_at`, `kind: operator_reference`, `spatial_support: longitude_mean`.
Describe comparable conditions; a source label does not validate the reference.
For Prophet mode, omit per-point reference values and use `reference: null`.

Density/background are longitude means. Longitude is provenance, not resolved
local density. Prophet background uses matching geometry/time/solar inputs and
fixed low geomagnetic activity: a model counterfactual, not observed climatology.
Older releases and DTM2020 lack it. Missing references remain `not_assessed`.

## Operator output

Each saved assessment records input revision, release identities/times, sampled
density/reference/ratio, warning location/span, peak ratio, reasons and data age.
No extrapolation beyond the supplied grid; missing samples/large gaps break intervals.
Single samples establish point warnings, not continuous exposure or precise crossings.

| Attention | Meaning |
| --- | --- |
| `within_reference` | Current coverage below configured ratio rules; not a safety guarantee |
| `review`, `high` | Configured ratio rule reached |
| `watch` | CME arrival context |
| `not_assessed` | Missing/invalid reference or insufficient usable data |

Stale density, unavailable CME context, expired trajectories, edits and pauses
are explicit data statuses. CME overlap is an association; it neither proves
causation nor multiplies density. Arrival spreads are uncalibrated. Unavailable
CME context does not establish absence of incoming events.

Journal entries record attention/data transitions and link exact source runs.
Polls are deduplicated; acknowledgment records review only. Missing-data periods
retain the known attention baseline. Accounts see only their own data.

## Service contract

Proxy prefix: `/api/v1/dashboard/density-advisory`; direct Uvicorn omits `/api/v1`.
Session determines owner; writes require configured dashboard Origin.

| Method / suffix | Result |
| --- | --- |
| `GET /`, `POST /` | Compact list/create profile; maximum 100/account |
| `GET /{id}` | Full owned inputs |
| `PUT /{id}?revision=N` | Replace inputs; optimistic revision check |
| `POST /{id}/assess` | Assess/reuse matching profile and releases |
| `GET /{id}/history?limit=20` | Run identities/times; maximum 50 |
| `GET /{id}/runs/{run_id}` | Exact immutable snapshot |
| `GET /{id}/runs/{run_id}/inputs/{product}` | Retained release: `atmospheric-density` or `solar-wind-events` |
| `GET /events?page=1` | Journal; 20/page |
| `POST /events/{event_id}/ack` | Idempotent review acknowledgment |

Worker polls every 60 seconds, fetching releases once/cycle. Matching runs are
reused; invalid artifacts retry after five minutes. Freshness changes are evaluated
on reuse. Locks/revision checks and atomic run/journal/input writes prevent partial
publication; one profile failure does not block others.

New runs retain exact input releases in Intelligence DB, deduplicated by ID and
checked by SHA256. Retained reads never contact Prophet. Absent products, another
owner's data and old unretained runs return 404; older runs are not backfilled.
Numerical replay also requires the matching private core/configuration.

All calculations and attention policy live in private `intelligence-core`.
[Operation](../apps/intelligence/README.md) · [Experimental loss monitor](leo-drag.md)
