# Solar wind aggregation, version 1

Use the [read-only audit](aggregation-audit.md) to compare saved results with
retained raw data and estimate history older than the proposed retention period.

Apply migrations through `20260908_0008`, then run `pnpm app:aggregate-solar-wind`.
Production cron invokes the same command every five minutes. Each invocation
processes at most 240 source hours (`--limit` overrides this); repeat to drain a
larger backlog. This is independent of ingestion and forecasting. No raw records
are deleted. Kp/Dst retain their original intervals.

`solar_wind_aggregate` stores UTC-aligned five-minute and hourly buckets separately
for magnetic and plasma data. The primary key is kind, resolution and bucket start.
Each row carries an aggregation version, calculation time, and JSON statistics.
See the [history API](solar-wind-api.md#api) for resolution choices and range limits.

## Responses and coverage

Aggregate history returns mean, minimum, maximum, counts, coverage and source
changes. Completed, current-version buckets remain visible while recalculation
is pending, with `recalculation_pending` on their points. Reads do not calculate
data. `evaluated_from` rounds the left edge down to a UTC bucket boundary;
`evaluated_to` rounds the right edge down to a closed boundary. Statistics cover
these full windows.

`coverage` uses native minute counts in available calculated windows only. It
excludes windows with no saved result: coverage there is unknown, not zero. A
separate `processing` object reports available/expected windows, unavailable
windows and windows awaiting recalculation. Coverage while recalculation is
pending describes the last calculated data, which may be revised. Gap ranges
identify partially covered calculated windows, not exact missing-minute locations.
Graphs connect adjacent means, break missing buckets and spacecraft transitions,
and show min/max and coverage only in the tooltip.

## Stored statistics

Per metric: valid count, full-bucket expected count, absent-minute count,
unusable-minute count, coverage percentage, sum, mean, minimum, maximum, and
first/last valid values with timestamps. Empty means and extrema are null.
Bz additionally records negative sample count, which is **not** a continuous
southward duration. Provider-flagged values and missing values are excluded.
Hourly statistics are calculated directly from raw minutes, weighted equally.

Only active spacecraft are selected. At equal timestamps the newest receipt wins,
then the alphabetically first spacecraft, matching raw history selection. If there
are multiple timestamps in a UTC minute, the latest is used once. This minute-slot
rule differs from the raw API, which preserves distinct native timestamps.
`source_sequence` and `source_changes` record switches within a bucket.
`max_gap_minutes` is the largest internal gap between selected minutes; it excludes
edge gaps and metric-specific invalid values. No interpolation is performed.

Only closed windows are calculated. `window_complete` is retained as true for
compatibility with stored results. It describes elapsed time, not data completeness.
`evaluated_through` is the window end.
Expected counts always refer to the full bucket (5 or 60 minutes).

## Recalculation

A PostgreSQL trigger enqueues affected source hours atomically with raw inserts,
updates and deletes, including active-spacecraft changes. Migration 0007 queues
existing source hours.
The worker locks each job and commits aggregate updates and job removal together.
Queue conflict updates lock the row, preventing concurrent ingestion from losing
a recalculation request while the worker removes a job. Worker failures roll back
without acknowledging the job. Unchanged closed-bucket recalculation preserves its
calculation timestamp. A current source hour stays queued until the hour closes;
each run processes that job at most once, saving only already closed five-minute
windows. The final hour is calculated even if the collector has stopped. Jobs survive process
restarts. Migration 0008 requeues legacy partial results once so they are replaced
when their windows close. Stop older aggregation workers before switching versions.

A source hour with observations produces all elapsed buckets, including empty
five-minute buckets. Entire hours with no observations and no queued changes can
have no aggregate rows; their observation coverage is unknown.
Changing the aggregation version requires explicitly requeuing historical hours
whose raw data are retained. [Retired hours](solar-wind-cleanup.md) cannot be
requeued or rebuilt without restoring their raw data.

## Checks

```sh
./scripts/test-python -q
PYTHONPATH=apps/api apps/api/.venv/bin/python apps/api/tests/integration/verify_solar_wind_aggregation.py
```

The integration check uses a temporary PostgreSQL schema and validates the
migration, queue, revisions, rollback, idempotency, recovery and downgrade.
