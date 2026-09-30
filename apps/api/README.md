# API

FastAPI serves public observations/forecasts, dashboard authentication and usage
statistics. It owns only dashboard storage; Clio and Prophet reads go through HTTP
with no SQL, provider-download or live CSV fallback. Model evaluation metrics are
static deployed artifacts. See [setup](../../README_DEPLOY.md#local-development) and
[commands](../../docs/commands.md).

## Configuration and access

Configure `API_DB_*`, `OBSERVATIONS_URL`, `OBSERVATIONS_SERVICE_TOKEN`,
`FORECASTS_URL`, `FORECASTS_SERVICE_TOKEN`, `DASHBOARD_ORIGINS` and
`DASHBOARD_COOKIE_SECURE`. Service tokens authenticate internal HTTP requests;
they are separate from dashboard user sessions. `./argus api user --help` lists
user-management actions.

The running `/docs` is the request/response reference (externally `/api/v1/docs`).
Public observation routes use `{success,data,error}` and `Cache-Control: no-store`.
Empty storage returns empty series/null values, never invented zeros; dependency
or storage failures remain HTTP errors.

## Compact response contracts

Forecasts (public and private), forecast metrics, atmospheric density, LEO drag,
solar-wind/geomagnetic latest and history, and the observation summary return
compact data by default. Add the boolean query parameter `?meta=true` to include
typed descriptions in `data.meta`. For the drag POST, this is a query parameter,
not a request-body field. Omitted or `meta=false` means the block is absent, not
null. Expanding metadata never changes the shape or values of the main data.
Missing measurement values remain explicit nulls in either mode.

| Response | Optional metadata location |
| --- | --- |
| Forecasts and evaluation metrics | `meta.variables.<variable>.unit`, once per variable, not per forecast point |
| Solar wind / geomagnetic | `meta.series.<metric>`: labels, units, coordinate frames, sources and processing descriptions |
| Observation summary | `meta.solar_wind`, `meta.geomagnetic`, `meta.changes_1h`, `meta.southward_bz`, `meta.lookback_minutes` |
| Atmospheric density | `meta.model`, `meta.history_start`, `meta.background_method`, `meta.background_interpolated_days`, `meta.dtc_method` |
| LEO drag | `meta.model` and `meta.source` with the density methodology above |

For example, `/public/forecasts/solar-wind-speed?meta=true` adds
`"meta":{"variables":{"v":{"unit":"km/s"}}}` inside `data`; individual
predictions have no `unit` field in either mode.

Units and coordinate frames are fixed contracts, not selectable response options:

| Observation variable | Unit / frame |
| --- | --- |
| `v` | km/s |
| `n` | cm^-3 |
| `t` | K |
| `bx`, `by`, `bz` | nT, GSM |
| `bt`, `dst` | nT |
| `kp` | Dimensionless Kp index |

Summary changes use the corresponding variable's units; southward-Bz duration is
in sampled minutes. Forecast units are listed in each endpoint's OpenAPI product
table. Density and drag fields carry units in their names (`rho_kg_m3`,
`altitude_km`, etc.); density no longer repeats a separate `unit` field.

Always present where applicable: quality, freshness and stale thresholds, native
interval boundaries, coverage/gaps, aggregate resolution and pending processing,
spacecraft transitions, risk thresholds/reasons and assumptions. Density and drag
retain `driver_mode: observed_persistence` and source timestamps; the boolean
`background_interpolated` identifies gap-filled backgrounds even without metadata.
Drag's compact `source` also retains `release_id`. Detailed interpolation dates
are available in metadata. Raw provider flags and aggregate `negative_count`
remain internal; the public API exposes normalized `quality`.

Hourly normalized `/observations/latest` and `/observations/history` already
contain only data and have no metadata expansion. `/observations/status` and
dashboard monitoring are dedicated diagnostic contracts and keep their detail.
Internal Clio/Prophet/Intelligence contracts keep the provenance and validation
needed by computation; the API uses separate public projections.

This is an in-place breaking change for the pre-client API. The web observation
views use fixed units/labels and compact responses. The LEO panel explicitly
requests metadata to show its density model. Deploy API and web together.

## Observation reads

Routes below are relative to `/api/v1/public/observations`:

| Routes | Semantics |
| --- | --- |
| `/solar-wind/latest`, `/solar-wind/history` | Native L1 observations or stored aggregates; optional `metrics` defaults to `bx,by,bz,bt,v,n,t` |
| `/geomagnetic/latest`, `/geomagnetic/history` | Three-hour Kp and hourly Dst; native interval boundaries |
| `/summary` | Observation changes and southward-Bz duration, not forecasts |
| `/status` | Source collection progress and data freshness, not process liveness |

History dates must be timezone-aware. Solar wind selects timestamps in `[from,to)`:
`1m` allows 7 days, `5m` 31 days, `1h` 366 days. `auto` uses minutes through 24h,
five-minute buckets through 7 days, then hourly buckets. Defaults are 24h and `1m`.
Aggregates include only closed UTC windows and are never calculated during reads.
Unknown metrics return 422. Gaps remain missing timestamps; invalid values are null.
Latest never substitutes an older good value for a missing newest sample.

Geomagnetic history defaults to 24h, allows 31 days and selects full intervals
overlapping `[from,to)`. Adjacent queries can repeat an interval; deduplicate by
`metric, interval_start`. Completed intervals are not necessarily scientifically
final. See [Clio](../clio/README.md#sources-and-storage) for sources, quality and freshness.

## Summary rules

Hourly changes compare two five-minute means one hour apart, anchored to the
metric's latest sample. Each window needs 4/5 valid samples and the last hour
48/60. Old inputs (>10 minutes), flagged/missing newest data, spacecraft changes
or insufficient coverage suppress the result; the response reports reasons and coverage.

Southward Bz counts consecutive valid negative minute samples from one spacecraft.
A preceding nonnegative sample establishes onset; a latest zero gives zero duration.
Gaps, flags or spacecraft changes before a known onset make the result unavailable.
If the entire available sequence is negative, `lower_bound` indicates only the
observed duration. Lookback is 75 minutes; sampled minutes are not proof of
continuous real-time duration. Summary values are statistics, not impact estimates.

## Project monitoring

Administrators see service health, per-source/per-metric observation timestamps,
per-product forecast publication/generation status, host resources and separate
API/site traffic on `/dashboard`. Clients have a separate placeholder landing
page. See [project monitoring](../../docs/project-monitoring.md) for permissions,
configuration, collection semantics, retention and deployment migrations.

## LEO drag

`POST /public/risks/leo-drag` proxies circular-orbit drag assessments to
Intelligence. See [request example and limits](../../docs/leo-drag.md).
