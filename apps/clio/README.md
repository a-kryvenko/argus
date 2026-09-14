# Clio runtime

Clio owns observations, collection status, aggregation/retention and their
PostgreSQL migrations. The public `packages/clio` library remains the provider
fetch/parse layer. The installed `argus-clio` application adds storage and
operational entry points. Solar wind and Kp/Dst remain separate processes.

## Setup

```bash
uv sync --project apps/clio --frozen
uv sync --project apps/api --frozen
```

The private forecast-core checkout is still required for Clio ingestion's
calibrated solar indices, historical density drivers and legacy model-input
normalization. These calls are isolated in `services/calibration.py` and loaded
only by ingestion. The HTTP read process does not import the private backend.
Private algorithms are not copied into public code. API no longer installs the
provider library or private backend.

Configure the following locally in `.env.local` (production passes explicit
variables to each container, without sharing the entire environment file):

| Variable | Consumer |
| --- | --- |
| `DB_NAME`, `DB_HOST`, `DB_PORT` | Database connection location; local defaults host=localhost, port=5432 |
| `DB_USER`, `DB_PASSWORD` | Operator bootstrap only; a separate database administrator |
| `API_DB_PASSWORD` | API runtime, fixed role `argus_api` |
| `API_MIGRATION_PASSWORD` | API migrations, fixed role `argus_api_migrator` |
| `CLIO_DB_PASSWORD` | Clio runtime, fixed role `argus_clio` |
| `CLIO_MIGRATION_PASSWORD` | Clio migrations, fixed role `argus_clio_migrator` |
| `OBSERVATIONS_URL` | API and Prophet; locally `http://127.0.0.1:8001`, production `http://clio:8000` |
| `OBSERVATIONS_SERVICE_TOKEN` | Shared read-service credential for Clio, API and Prophet |

Use distinct strong random passwords. Runtime code has no fallback to `DB_USER`
or `DB_PASSWORD`, and no environment variable can select an administrative role.
The four domain role names and the `api`/`clio` schemas are reserved for this app.

For a new database, or when adopting the legacy database after stopping all
application writers:

```bash
./scripts/domain-db                 # validate and roll back
./scripts/domain-db --apply         # provision/adopt, commit
./scripts/clio migrate upgrade head
pnpm db:migrate
```

Read [database ownership and provisioning](../../docs/domain-storage.md) before adopting existing data.

## Commands

```bash
./scripts/clio serve --host 127.0.0.1 --port 8001
./scripts/clio collect solar-wind --watch
./scripts/clio collect geomagnetic --watch
./scripts/clio refresh
./scripts/clio aggregate --limit 240
./scripts/clio schedule refresh
./scripts/clio schedule aggregate
./scripts/clio audit --help
./scripts/clio cleanup --help
./scripts/clio check-health solar-wind
./scripts/clio migrate current
```

Production Compose runs `clio`, `solar-wind`, `geomagnetic`, `clio-refresh` and
`clio-aggregate`. Scheduling is owned by Clio: hourly refresh at :00 UTC and
aggregation every five minutes. No application commands remain in crontab.

`clio.scheduled_job` records the latest completed slot. Failed attempts do not
advance it; the process retries every 60 seconds. Restarts catch up the latest
due slot, without replaying every missed hour. A session advisory lock shared by
manual and scheduled refresh/aggregate calls prevents concurrent supported runs
of the same job. Existing per-source transaction locks and aggregate row locks
are preserved. Manual calls do not advance scheduled completion markers.

Work remains idempotent/retriable rather than exactly once: a crash after writes
but before recording completion can repeat a job. Session locks also require a
live database connection; the underlying source/upsert and aggregate row-lock
rules remain the final consistency boundary. SIGTERM allows the current job to
finish within the Compose stop grace period. History deletion is never scheduled.

## Read interface

All data routes require `Authorization: Bearer <OBSERVATIONS_SERVICE_TOKEN>`:

- `/internal/v1/observations/latest`, `/history`, `/status`, `/summary`
- `/internal/v1/observations/solar-wind/latest`, `/solar-wind/history`
- `/internal/v1/observations/geomagnetic/latest`, `/geomagnetic/history`
- `/internal/v1/observations/browse` — paginated raw/normalized dashboard data
- `/internal/v1/observations/forecast-inputs` — the unchanged Prophet contract

Query parameters and response envelopes match the corresponding existing public
and dashboard routes. The public API retains user authorization and forwards
only the defined reads. It returns 503 when Clio is unavailable, with no SQL or
provider fallback. The API's own login/user/statistics storage stays local.

`/health/live` checks the HTTP process; `/health/ready` checks that Clio storage is
available and migrated. These probes contain no observation data. Collector
healthchecks retain their independent per-container heartbeat files. Clio is not
published through nginx; consumers reach it on the Compose observations network.
