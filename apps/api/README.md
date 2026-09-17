# API

FastAPI serves public observations/forecasts, dashboard authentication and usage
statistics. It owns only dashboard storage; Clio and Prophet reads go through HTTP
with no SQL, provider-download or live CSV fallback. Model evaluation metrics are
static deployed artifacts. See [setup](../../README.md#local-development) and
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
