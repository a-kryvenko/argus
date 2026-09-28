# Clio

Clio owns source ingestion, immutable historical recovery, native sensor data,
normalization, file artifacts and the observations HTTP API. Python package:
`apps/clio/src/clio`. PostgreSQL schema: `clio`.

## Commands

```bash
./argus clio collect                       # All configured live observations
./argus clio collect bx by bz v n t kp dst
./argus clio collect aia193
./argus clio backfill                      # Per-observation configured depth
./argus clio backfill v n t --from 2026-08-04 --to 2026-08-05
./argus clio normalize                     # Stored measurements only
./argus clio aggregate
./argus clio status
./argus clio audit --json
./argus clio cleanup --json                # Preview; --apply performs cleanup
./argus clio migrate
```

Metrics are positional, separated by spaces. `fetch-live`, `--metrics`, the old
`collect aia/solar-wind/geomagnetic`, and `refresh` interface were removed.
`serve`, `worker`, `schedule` and `check-health` are operational commands.
Manual commands use the worker's locks but do not advance schedule markers.

## Layout and responsibilities

- `commands/`: thin command handlers and output, no provider selection rules.
- `scheduling/`: periodic execution, PostgreSQL locks and completion markers.
- `ingestion/`: source registry, adapters, selection and monitoring-independent orchestration.
- `providers/`: public provider transports and parsers; GONG is retained.
- `observations/`: metric schema, live/backfill persistence, normalization and calibrated solar indices.
- `domains/geomagnetic.py`: geomagnetic index intervals and persistence.
- `domains/aia/`: immutable originals/receipts, artifact recovery, FITS extraction and stored features.
- `domains/solar_wind/`: native RTSW data, aggregation, history, audit and retention.
- `monitoring/`: source health, heartbeat and collection status.
- `routers/`, `db/`, `migrations/`: HTTP contracts, persistence and migration history.

The worker uses one scheduler process. Every task, including numeric live and
native RTSW collection, runs in a temporary executor. Three independent live
lanes and at most two background tasks can run concurrently; a lane never
overlaps itself. Background tasks are ordered by waiting time. The scheduler
retains advisory locks until executors exit and records completion from their
reports. Provider libraries and working data are released on executor exit.
Shutdown stops dispatch, drains active tasks for up to 570 seconds, then
terminates remaining executors before releasing their locks. Starting an
interpreter for each run trades CPU/startup time for lower idle memory.
Standalone `schedule` commands remain available for diagnostics.

Native RTSW and metric observations are distinct products. Do not mix spacecraft
sample timestamps, propagated timestamps, coordinate frames, or aggregate windows.
Reads do not fetch providers. Model calibration is accessed through
`observations/calibration.py`; HTTP reads do not load the private backend.
Training-only loaders, annual OMNI export and notebook tests live under
`scripts/training`, outside the application runtime.

## Collection and backfill

`configs/project.yaml` defines `clio.observations`: each observation has ordered
`sources.live` and `sources.historical`, schedules and `backfill.days`.
Live collection groups shared source requests, checks freshness and falls back per
metric. Backfill requests missing slots only and inserts with conflict-do-nothing;
existing values, provenance and first receipt times are preserved. Normalized
rows are derived and may be recomputed. Unpublished archive dates remain gaps,
not source failures. Unknown/network failures remain visible in reports.

Historical plasma priorities are OMNI, ACE, SOHO/CELIAS. SOHO uses five-minute
proton moments from https://l1.umd.edu/. It converts each thermal speed to kelvin
using `mp*(Vth*1000)^2/(2*kB)` before hourly averaging and requires six valid
samples per metric/hour. Nonpositive and nonfinite samples are excluded. Annual
ZIP downloads are reused within one invocation. SOHO timestamps are spacecraft
measurement times, with no Earth-propagation correction or cross-instrument
calibration; SOHO only fills missing observations.

AIA live uses `aia.nrt_193`; historical backfill uses `aia.synoptic_193`.
Live checks the current hour and previous three hours. Defaults: hourly live,
six-hour backfill, 40-day history (maximum 60 days). Originals and receipts live
in `data/observations/aia193` (`ARGUS_AIA_ARCHIVE` overrides the root).
Missing originals must match the recorded SHA256; missing/corrupt derived caches
are rebuilt from originals. Recovery retains the source and first availability.
QC-rejected originals remain rejected. NRT allows only the additional QUALITY
bit 30 (`Q_NRT`); archive quality remains strict. Per-slot locks serialize recovery.

## Worker

Seven supervised processes run independently:

```text
schedule native-wind
schedule live --kind numeric
schedule normalize
schedule aggregate
schedule live --kind file
schedule backfill --kind numeric
schedule backfill --kind file
```

Native collection retains its own cadence. Numeric and file schedules use separate
locks so AIA warmup does not block numeric live collection. Observation markers
remain `live.<metric>` / `backfill.<metric>`; existing normalization/aggregate
markers and lock IDs are preserved. Logs report metrics, status and gap counts.
`partial` can mean missing or QC-rejected data without transient source errors.

## Tests

```bash
PYTHONPATH=apps/clio/src .venv/bin/python -m pytest --import-mode=importlib apps/clio/tests
```

Tests are grouped by `providers`, `ingestion`, `observations`, `aia`, `solar_wind`,
`scheduling`, `cli`, `api`, and `integration`. Protect behaviour: units, quality,
source fallback, freshness, insert-only backfill, artifact checksums, concurrency,
API contracts and scheduler retries. Avoid preserving tests of removed wrappers.

Integration tests require `TEST_DATABASE_ADMIN_DSN` pointing to an isolated test
PostgreSQL server; otherwise pytest skips them. They create temporary schemas.
Research/notebook tests are separately run from `scripts/training/tests` and may
require local notebooks/private calibration dependencies.
