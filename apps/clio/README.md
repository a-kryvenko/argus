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
`serve`, `worker` and `check-health` are operational commands.
Manual commands use the worker's locks but do not advance schedule markers.

## Layout and responsibilities

- `commands/`: thin command handlers and output, no provider selection rules.
- `scheduling/`: periodic execution, PostgreSQL locks and completion markers.
- `ingestion/`: source registry, adapters, selection and monitoring-independent orchestration.
- `providers/`: public provider transports and parsers, including GONG live/archive.
- `observations/`: metric schema, live/backfill persistence, normalization and calibrated solar indices.
- `domains/geomagnetic.py`: geomagnetic index intervals and persistence.
- `domains/gong.py`: immutable magnetograms, private feature extraction and stored feature reads.
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
Use one-shot commands for manual collection, backfill, normalization and aggregation.

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

`clio worker` dispatches seven task families into temporary executors:
native RTSW collection, numeric live collection, file live collection,
normalization, aggregation, numeric backfill and file backfill.
The worker owns all periodic loops; these are internal tasks, not CLI commands.

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

## Forecast inputs

`GET /internal/v1/observations/forecast-inputs?as_of=...` returns the versioned
`common.schemas.forecast_inputs.ForecastInputs` contract in one read-only database
snapshot. It includes normalized observations, raw density drivers, observed speed,
AIA features, native hourly wind and optional `gong` features. Reads do not collect.

GONG is configured as a file observation: `collect gong` and `backfill gong` use
`gong.live` and `gong.archive`, with hourly live collection and a two-day backfill
by default. One original per UTC hour is retained in `clio.gong_snapshot`, together
with compressed FITS bytes, SHA256, first receipt, provider URL and `gong-bands-v1`
features. Conflicts never overwrite earlier receipts or originals. Feature
extraction uses the lazy private `forecast_core.observations` boundary during
collection; HTTP reads use stored JSON only. Apply `20260929_gong_snapshot` first.
The latest frame whose observation and receipt are both at or before `as_of` is
returned; Prophet separately enforces the configured age limit. Originals have no
automatic retention policy.
## Hourly SDO observations

Clio collects individual hourly images for eight AIA wavelengths (94, 131, 171,
193, 211, 304, 335, 1600) and HMI LOS. Each archived image is 512x512 and has
its own observation time and first-receipt metadata. Selection of model inputs
and temporal assembly belong to private forecast-core, called by Prophet.

`clio.providers.sdo_images.download_observation(url, channel, slot=..., now=...)`
downloads a scientific FITS with bounded size/time and checks its instrument, units,
resolution and observation time. HMI TAI times are converted to UTC. Stale
`current` files are rejected. Both 1024x1024 NRT and 4096x4096 scientific images
are accepted, with source resolution preserved in metadata. These source arrays
are labelled `fits-unregistered-v1`, not calibrated encoder inputs.
This loader does not calibrate/reproject FITS or invoke reconstruction.

`common.sdo_images` provides atomic 512x512 observation writes, first-receipt
metadata, causal reads and `prune(root, now=...)` for the current UTC hour plus
143 previous hours. The archive accepts only the nine observed channels and
exposes individual images through `read_image`. Files live at
`<data_root>/observations/sdo/YYYYmmddTHH0000Z/<channel>.npz` (or
`ARGUS_SDO_ARCHIVE`). Each NPZ contains `image` (float32) and `metadata` (JSON).

The worker enables collection through `clio.sdo_images.enabled` in project.yaml:

- `sdo-live`: on startup and every 300 seconds after completion, checks the
  current hour and three preceding hours to allow for source publication delays.
- `sdo-warmup`: starts with the worker and fills the remaining retained window
  in six-hour batches, with 60 seconds between batches. A persistent cursor
  traverses gaps even when upstream files are missing, then repeats the window.
- `sdo-cleanup`: on startup and every 3600 seconds, deletes expired hourly files.

All lanes have independent process locks; warmup uses the bounded background
pool and cannot occupy live worker capacity. Downloads use four threads per
collection lane. Existing images are immutable and skipped. HTTP 404 is an
unavailable observation; other failures are logged and retried on subsequent
passes. Warmup is best effort: source gaps remain explicit, with no substitution.
Retention includes the current UTC hour and previous 143 hours; expired files
are removed on the next cleanup pass. Publication rejects expired slots.
Manual cycles: `clio sdo-images live`, `clio sdo-images warmup`,
`clio sdo-images cleanup`. These honor the same enable flag and locks.

The collector reduces public 1024x1024 numeric FITS to 512x512 using finite-only
area means (`sdo-area-mean-unregistered-v1`). Blocks without measurements remain
NaN, notably outside the HMI disk. AIA remains in source DN and HMI in Gauss.
Receipts retain source headers, exposure, quality, source checksum and true UTC
observation/receipt times; pixel scale and reference pixel in the reduced header
are adjusted for the new grid. This is **not** Surya-normalized or co-registered
encoder input. Training must explicitly use this observation convention or add
an agreed preprocessing pipeline; historical normalized-before-reduction inputs
are not numerically interchangeable with these observations.

`GET /internal/v1/observations/sdo-images` returns a JSON catalog protected by
`OBSERVATIONS_SERVICE_TOKEN` (Bearer). Optional `start` and `end` are aware whole
UTC hours, with an exclusive end and a maximum range of 144 hours. Optional
`channel` selects one observed channel. `as_of` defaults to now and excludes
observations received later; the default range is the 144 slots ending in its
hour. The successful response envelope contains `data.items` and `data.missing`.
Each item has an absolute `path` to its NPZ, `shape: [512, 512]`,
`dtype: "float32"` and its `metadata` receipt. No image arrays or download URLs
are sent. `metadata.sha256` identifies the original FITS, not the NPZ file.

Clio and Prophet currently mount the same `./data` at `/var/www/data`, so these
paths are directly readable in both services. Any `ARGUS_SDO_ARCHIVE` override
must also resolve to shared storage at the same path. Catalog reads only inspect
receipts and never download, transform or assemble images. Missing observations
are explicit; an empty archive returns an empty `items` list. Paths are valid at
read time and do not prevent retention from deleting files later. A consumer
must handle a file disappearing before it opens it.

The nine-channel NRT sources were confirmed on 2026-09-30 at jsoc1's
`data/aia/synoptic/nrt/` and `data/hmi/fits/` (1024x1024 FITS).
Calibration, registration, model normalization and temporal assembly remain
separate forecasting/training concerns; Clio does not assemble pairs.
