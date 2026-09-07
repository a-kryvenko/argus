# Minute solar wind observations

The `/live` page uses unpropagated NOAA SWPC RTSW observations at L1. Magnetic
components are GSM; total field, proton speed, density and temperature are scalars.
Source documentation: https://www.spaceweather.gov/products/solar-wind

The source files contain one-minute samples for multiple spacecraft over the last
24 hours. Argus retains every spacecraft record and the source `active` selection.
Public responses use NOAA's active spacecraft independently for magnetometer and
plasma data. They never substitute an inactive spacecraft when the active stream
has missing or flagged measurements.

## Collection and storage

```bash
pnpm db:migrate
pnpm app:solar-wind          # one pass; nonzero exit on source failure
pnpm app:solar-wind --watch  # repeat every 60 seconds; retry failures
```

Deployment runs `app.commands.collect_solar_wind --watch` in its own Compose
service. Each source has a separate transaction and advisory lock. One source's
failure does not roll back the other source. Overlapping collectors skip a locked
source. A failed continuous cycle is logged and retried without clearing stored data.

`solar_wind_observation` stores `(kind, observed_at, spacecraft)`, source selection,
parsed nullable values, the original source record (including quality flags), and
`received_at`. This timestamp is the receipt of the latest changed source record,
not the most recent polling time. Repeated identical records are no-ops. Source
corrections replace the stored version; revision history is not implemented yet.
No data is automatically deleted in this stage. Backfill after downtime is limited
to the source's rolling 24-hour window; longer outages remain visible as gaps.

This storage is separate from `measurement` and `normalized_observation`. Existing
hourly ingestion and normalization continue to supply forecasts with their existing
propagated, potentially gap-filled input. Minute collection never invokes them.

## API

Paths below are relative to `/api/v1`. Existing hourly endpoints are unchanged.

- `GET /public/observations/solar-wind/latest?metrics=bz,bt,v,n`
- `GET /public/observations/solar-wind/history?from=2026-09-07T00:00:00Z&to=2026-09-08T00:00:00Z&metrics=bz,bt,v,n`

Both return the standard `{success,data,error}` envelope. `metrics` is optional;
all seven metrics (`bx,by,bz,bt,v,n,t`) are included by default. Unknown metrics are
rejected with 422. `history` defaults to the last 24 hours and accepts timezone-aware
`from` and `to`, with a positive range of at most seven days. Intervals are
`[from,to)` so adjacent requests do not duplicate endpoints. Longer stored history
can be retrieved with adjacent requests; the API never silently truncates a range.

`data.series` is keyed by metric. Each series includes its label, units, source URL,
coordinate system, measurement location, resolution and time basis. `propagated`
is always false. The one-minute source aggregates are retained as provided; Argus
performs no averaging, interpolation, rounding of timestamps or gap filling.

For `latest`, each series contains:

- `latest`: sample or null, with `observed_at`, `received_at`, `spacecraft`, `value`,
  `quality` and the source's `provider_quality` (`overall_quality`).
- `age_seconds`: age of the measurement at `generated_at`, independent of polling.
- `status`: `missing` if no value exists, `stale` after 600 seconds, otherwise `fresh`.
  Freshness does not imply good quality.

For `history`, each series contains `points` in ascending measurement-time order.
Omitted timestamps are gaps; explicit unavailable/invalid values are null. The
latest sample may also contain null and never falls back silently to an older value.

`quality` is `missing` for a null/invalid numeric value, `flagged` for a nonzero
provider overall quality, otherwise `unverified`: Argus has not independently
validated it. All original provider flags remain in the stored source record.
Flagged numeric values are preserved in the API, but omitted from chart lines.
Charts also break at missing minutes and spacecraft changes. UTC timestamps and
synchronized cursors allow comparing the three charts. Hourly indices remain in
a separate expandable section with their normalization notice.

No database measurements is a successful empty result (null latest samples or empty
point lists), not an invented zero. Database failures remain HTTP errors. Responses
use `Cache-Control: no-store`; the page polls every minute and retains previously
loaded data with an error notice when refresh fails.
