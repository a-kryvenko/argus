# Clio

Clio owns observations, normalization, aggregation, collection diagnostics and
scheduled ingestion. `packages/clio` provides public provider parsing; ingestion
owns GOES/GFZ history downloads and the JB2008 history cache, and uses private calibration through `services/observations/calibration.py`. HTTP reads do not
load the private backend. See [setup](../../README_DEPLOY.md#local-development) and
[commands](../../docs/commands.md).

## Service layout

`src/argus_clio/services/` groups related operations by domain:

- `solar_wind/`: native observations, aggregate calculation (`aggregation.py`),
  aggregate reads (`history.py`), consistency checks (`audit.py`) and retention.
- `aia/`: image collection, archive location and forecast feature reads.
- `observations/`: normalized model inputs, shared persistence, backfill,
  solar-index calibration and density history.
- `collection/`: source specifications, attempt status, heartbeat and monitoring.

`geomagnetic.py` handles native Kp/Dst. `summary.py` combines source summaries;
`coverage.py` calculates coverage for both solar wind and geomagnetic history.
Import the concrete modules; package initializers do not re-export services.

## Configuration and interfaces

Use `CLIO_DB_*` for Clio's database and `OBSERVATIONS_SERVICE_TOKEN` for internal
HTTP authentication. Consumers configure `OBSERVATIONS_URL`.

Routes under `/internal/v1/observations` require a bearer service token:
`latest`, `history`, `status`, `summary`, `solar-wind/latest`, `solar-wind/history`,
`geomagnetic/latest`, `geomagnetic/history`, `browse`, `forecast-inputs`.
API forwards these reads with its own authorization; there is no SQL/provider fallback.

`forecast-inputs?as_of=<aware timestamp>` returns normalized observations for
`[as_of - 30 days, as_of]`, raw density measurements for `[as_of - 88 days, as_of]`,
`as_of` and `read_at`. Both sets share a repeatable-read, read-only transaction.
More than 250,000 raw measurements fails explicitly; statements time out after
60 seconds. `read_at` is not a durable snapshot ID; Prophet archives the response.

## Sources and storage

| Source | Native data | Poll interval | Display freshness |
| --- | --- | --- | --- |
| NOAA RTSW | Minute L1 solar wind; GSM magnetic components | 60s | Measurement age ≤600s |
| NOAA planetary Kp | Estimated fractional Kp, three-hour intervals | 60s | Interval-end lag ≤4h |
| Kyoto Dst via NOAA | Realtime Dst in nT, hourly intervals | 300s | Interval-end lag ≤2h |

Solar wind stores each `(kind, observed_at, spacecraft)` and NOAA's active selection.
Magnetic and plasma streams select their active spacecraft independently; missing
active data never falls back to inactive data. Geomagnetic records use
`(metric, interval_start)`. Receipt time tracks changed records, not polling.
Repeated records are no-ops; corrections replace previous values without revision history.
Downtime recovery is limited by each provider's rolling history. No automatic deletion runs.

Missing/nonfinite/sentinel values are null. Provider flags remain available;
flagged numeric values are retained but excluded from charts and derived summaries.
Kp with zero contributing stations is flagged. Other valid values are `unverified`,
not independently validated. Kp is estimated and Dst realtime; both can be revised.
Freshness and quality are independent. Native data retains gaps; normalized model
inputs are separate and may be propagated or filled.

## Scheduling and consistency

Production runs two containers: `clio` (HTTP) and `clio-worker` (`clio worker`).
The worker supervises the existing two collectors and two schedules in separate
child processes: hourly refresh at `:00 UTC` and aggregation every five minutes.
Their schedules, source locks and completion markers are unchanged; there is no
message broker, new job table or in-memory backlog. Collection stays independent
of normalization and aggregation. Failed scheduled jobs retry every 60 seconds. Restarts catch
up only the latest due slot. Manual jobs share locks but do not advance scheduled
completion markers. Source and job lock keys are defined in `db/locks.py` and do not overlap. Work is idempotent/retriable, not exactly once.

Run `./argus clio worker` in the foreground locally; `pnpm dev` also starts it.
Production Compose starts it automatically. Manual `refresh` and `aggregate`
commands run independently, using the same existing locks, without needing the
worker. Refresh and aggregation may run concurrently, as before.

If any child exits, even successfully, the worker stops its peers and exits with
an error so Docker can restart the service. Task failures handled by the existing
loops keep their normal retry behavior. SIGTERM/SIGINT stops new cycles, allows
active work to finish, and waits up to 570 seconds across all children before
killing and reaping remaining processes (Compose grace period: 10 minutes).

Each provider uses a separate transaction and advisory lock. Locked sources are
skipped; one source failure does not roll back another. Collection attempts commit
before fetching; observations and success commit together. Failure diagnostics
commit after rollback. Database outages require logs/healthchecks because the
unavailable database cannot record them. SIGTERM allows the active job to finish
within the configured stop grace period.

## Status and health

`./argus clio status` reports attempts, successful responses/saves, measurement age,
last error and consecutive failures. A retained error with zero consecutive failures
is historical. Polling unchanged data does not make the observation fresher.

Combined source status prioritizes: `collector_stalled` (unfinished >120s),
`collector_overdue` (no attempt for two poll intervals +60s), `collection_error`,
`collecting`, `source_delayed`, `data_unavailable`, `data_partial`, then `ok`.
A source without attempts is `not_started`. Collection and data status remain separate.

`/health/live` checks HTTP liveness; `/health/ready` checks migrated storage.
Collectors use separate heartbeat files inside the worker container.
`clio check-health worker` checks both collectors, every 30s after a 120s
startup period, with three retries. It checks collector progress, not scheduler
completion; scheduler failures remain visible in logs. Upstream errors alone do not fail liveness
while polling continues. One-shot collection does not write watch heartbeats.
Docker's unhealthy status alone does not restart a container.

## Aggregation audit

`pnpm app:audit-solar-wind` compares stored five-minute/hourly aggregates with raw
records over the last seven days. `--from` / `--to` accept whole UTC hours, at most
31 days per run; `--json` and `--detail-limit` control output.

The audit is repeatable-read/read-only and does not fetch, recalculate or delete.
It checks aggregation version, statistics, coverage and pending work; integer
counts are exact and float comparisons use 1e-9 relative/absolute tolerance.
Missing raw hours are unverifiable. Hours absent from raw, aggregate and queue
tables are outside scope; this is not proof of upstream completeness.

Exit 1 means issues or command failure; exit 0 can mean `ok` or `no_data`.
Details default to 200 while totals remain complete. Storage estimates scan the
entire raw table and identify records older than `--retention-days` (default 90).
They do not certify safe deletion or predict reclaimed disk space. Run off-peak;
statements have a 60s timeout. Cleanup must revalidate records when deleting them.

Provider-fetching notebooks live locally in `apps/clio/notebooks/` (ignored, as
were the original root notebooks). Training consumes the resulting datasets.
Only this domain imports the `clio` provider library; consumers use HTTP.


## Hourly AIA193 collection

The supervised worker now also runs `clio schedule aia` hourly (`:00 UTC`).
It requests **every hourly slot**, including01/02/03/04/05UTC, and retains original
FITS at `data/observations/aia193` (override: `ARGUS_AIA_ARCHIVE`). The normal mounted
Clio data volume must persist. The first run warms up40days, recent images first;
subsequent runs skip complete snapshots and retry missing slots within40days.
Missing404 observations are never filled. QC-rejected originals and reasons remain
on disk and are excluded from model inputs. Transient failures retry through the scheduler.
No image retention/deletion is added. Four download workers, bounded requests and files.

Run the database migration **before** starting the new HTTP/worker images:

```bash
clio migrate upgrade head
clio collect aia                 # one bounded40day warmup; also scheduled automatically
clio collect aia --history-days 1  # bounded manual collection, same advisory lock
clio schedule aia                # normally already part of clio worker
```

`aia_snapshot` records slot, actual observation time, actual received-at availability,
SHA256 and local paths. Historical downloads are available only from their real receipt;
we never backdate availability to an assumed2h latency. Forecast inputs select only
00/06/12/18UTC snapshots and temporal pairs, strictly received before as_of.
Clio derives sector/24h/rotation features from its own cache and returns them through
`/internal/v1/observations/forecast-inputs` as optional `aia_frames`. Prophet does not
read Clio disk. A40day bounded source read supports rotation history and the target-time
window; only10days of model features are sent. Archive persistence is separate from
model sampling, so changing the future model cadence does not require re-downloading.

## Historical backfill

An empty measurement table is bootstrapped from provider history. The legacy
`live_sensors.csv` bootstrap and its configuration setting are no longer used.

`./argus clio backfill --from 2026-08-31 --to 2026-09-09` restores missing
measurements in an existing database and rebuilds normalized observations in that
range. Dates mean midnight UTC; timezone-aware whole hours are also accepted.
`--from` is inclusive, `--to` exclusive, at most 31 days per invocation.

The command shares the refresh lock. OMNI supplies core observation metrics;
ACE supplies additional plasma values only (its GSE magnetic fields are not
substituted for GSM). The existing GFZ/GOES archive path supplies available solar
indices using configured calibration. Source failures are reported; available
sources can still be saved. No valid measurements from any source is an error.
Existing measurements and collection receipt timestamps are preserved. All database
changes commit together. Repeated runs can add newly available archive records.

The JSON result reports downloaded measurement count (including existing records),
normalized hour count, source errors and missing **observed** hours per core metric.
Normalization follows the existing interpolation policy; normalized rows do not
prove raw history completeness. Daily indices need not have hourly observations.
Check the reported speed gaps before retrying Prophet; archives may not yet cover
recent dates. Backfill does not generate forecasts or AIA images.
