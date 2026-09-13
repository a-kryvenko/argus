# Prophet readiness diagnostics: stage 3c.2a

## Deployment and scope

**No new environment variables, migrations or dependencies.** The existing
service tokens and database credentials remain required. Input diagnostics use
an additional `input_diagnostics` object in the existing run `provenance` JSONB.
Previously recorded snapshots are not rewritten; older runs report null diagnostics.

The one-time `import-schedule` step is removed from deployment because production
has already completed the database-scheduler cutover. Normal domain migrations
and service restart remain. For installations older than stage 3c.1, first follow
the explicit [scheduler cutover](prophet-scheduling.md).

This is diagnostic mode only. There are no new blocking thresholds or changes to
calculation eligibility, retry rules, publication or public forecast responses.
The existing atmospheric-density maximum age (6 hours by default, read from its
`max_age_hours` registry entry) remains enforced by the public API. Other products
explicitly report `unconfigured`, not healthy/current, until thresholds are agreed.

## Inspection

```bash
docker compose run --rm prophet prophet status solar-wind-speed
docker compose run --rm prophet prophet status atmospheric-density
docker compose run --rm prophet prophet show-run <run-uuid>
```

The corresponding internal endpoint is:

```text
GET /internal/v1/forecasts/{product}/status
Authorization: Bearer <FORECASTS_SERVICE_TOKEN>
```

`common.schemas.forecast_status.ForecastStatus` version 1 describes:

- The current release UUID, its originating run, model issue time and publication time.
- `release_age_hours`, measured from model `issue_time`, not from publication time.
- `freshness`: `unavailable`, `future_issue_time`, `unconfigured`, `stale`, or
  `within_age_limit`. The latter confirms the age check only, not overall validity.
- The latest applicable calculation attempt, its status/error and its artifact
  outcomes. An attempt that fails before creating artifacts is still visible.
- Saved diagnostics for both the current release's run and the latest attempt.

All metadata is read in one repeatable-read transaction, without decompressing
forecast payloads or running models. Unknown products return 404, invalid tokens
401, storage failures 503. A known product with no release returns a diagnostic
200 with `freshness=unavailable`. This endpoint does not change `/latest`, historical
release reads or service `/health/ready`, which still checks infrastructure readiness.

## Input evidence

New runs save diagnostics with their input snapshot. Ages are relative to the
snapshot's `as_of`, not the later time when an operator inspects it. The record
also preserves `read_at` and `thresholds_configured=false`.

Normalized observations report row count, first/latest timestamps, age, largest
gap between timestamps, duplicate/naive/future timestamps and missing or nonfinite
value counts. Missing optional S10/M10/Y10 values are counted separately and do
not imply every forecast product is invalid. Counts are evidence, not a substitute
for per-model feature/history requirements.

For each raw density input metric, the report includes count, latest source
timestamp, age and future measurement count. No universal sensor-age threshold
is assumed: daily solar indices and geomagnetic drivers have different cadence
and lag requirements. Density's existing preparation checks and skip reasons
continue to apply; diagnostics do not claim to replace them.

A recent normalized row does **not** establish freshness of each original sensor:
normalization and filling may preserve old source values. The response explicitly
sets `source_freshness_known=false`. Missing source timestamps and missing policies
must not be interpreted as a green readiness signal.

## Next decision: thresholds and enforcement

Stage 3c.2b requires agreed per-product limits for input age/history gaps and
forecast age, plus the response to breaches (wait/retry, skip a product, or reject
latest reads while preserving historical access). Collect the diagnostic evidence
first, distinguish normalization time from source time, and set those limits
explicitly. No unapproved numerical defaults or enforcement switch is introduced
by this release.
