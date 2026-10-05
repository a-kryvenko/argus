# Command reference

## Help

```bash
./argus help
./argus clio --help
./argus prophet --help
./argus intelligence --help
./argus demo --help
./argus api user --help
```

## Service processes

`./argus compose <arguments>` runs Docker Compose for the selected environment,
loading its env files. In dev:

```bash
./argus compose up -d --wait
pnpm dev                         # frontend, dev, in this terminal
./argus compose restart prophet # after editing worker code
./argus compose down
```

For the first startup, provision and migrate the databases before starting Python
services; see [local setup](../README_DEPLOY.md#local-development).

## Databases

| Command | Purpose |
| --- | --- |
| `./argus db provision [--apply]` | Preview database and owner creation; `--apply` executes the plan |
| `./argus db migrate` | Upgrade all four databases to `head`; stop on the first error |
| `./argus <service> migrate` | Upgrade one service database to `head` |
| `./argus <service> migration create -m "description" [--autogenerate]` | Create a migration file; development only |
| `./argus <service> migration current` | Show the current revision |
| `./argus <service> migration history` | Show migration history |
| `./argus <service> migration downgrade <revision>` | Downgrade to a revision; may delete data |

Services: `api`, `clio`, `prophet`, `intelligence`.
`--autogenerate` is available for API and Clio. Prophet and Intelligence use
handwritten migrations.

## Clio

Clio `collect`, `backfill` and `normalize` run once and exit.
Omitting metrics from `collect` or `backfill` selects all configured observations.
Metrics are positional and space-separated: `collect bx by bz`, without
`--metrics` or commas. The old Clio `fetch-live`, `refresh` and
`collect solar-wind/geomagnetic/aia` interfaces have been removed.

| Command | Purpose |
| --- | --- |
| `./argus clio normalize` | Normalize stored observations |
| `./argus clio worker` | Run collection, normalization and separate numeric/file live/backfill schedules |
| `./argus clio collect [METRIC ...]` | Fetch selected observations using configured live priorities |
| `./argus clio backfill [METRIC ...] [--from DATE --to DATE]` | Fill missing observations using configured history depths |
| `./argus clio check-health <solar-wind\|geomagnetic\|worker>` | Check collector health |

An empty database needs `collect` / `backfill` before `normalize`.

Clio manual commands use the same locks as the worker and do not advance its
scheduled completion markers. They also work when the worker is stopped.
Docker Compose starts the Clio worker in both dev and production.
The worker uses one scheduler and temporary executors with independent live
lanes and at most two concurrent background jobs. Without positional metrics, manual commands include all configured kinds.

## Prophet

Prophet `generate` (alias `refresh`) runs once; omitting the product selects
`all`. It attempts every selected product and publishes successes independently;
it exits with an error if any product fails. Workers handle routine generation
and verification; manual commands are for recovery or explicitly requested work.

| Command | Purpose |
| --- | --- |
| `./argus prophet generate [product]` | Generate and publish forecasts |
| `./argus prophet verify [product] [--days 7]` | Re-evaluate published releases against observations |
| `./argus prophet verification-report [product] [--days 7]` | Compute stored accuracy by artifact and model hash |
| `./argus prophet cleanup [--days 90] [--apply]` | Preview/delete old forecast history; preserve current releases and latest attempts |

Generation products: `solar-wind-speed`, `solar-wind-density`, `geomagnetic-activity`, `dst`,
`hmf`, `atmospheric-density`, `all`. Prophet generation calculates only the selected
product; `all` calculates every supported product using one observation snapshot.
`solar-radiation` is not supported by these commands.

`verify` and `verification-report` support the same products except
`atmospheric-density`; their default product is `solar-wind-speed`. `--days`
accepts 1–25 days and defaults to 7.

## Diagnostics

Clio and Prophet expose data status over authenticated HTTP:

| Route | Purpose |
| --- | --- |
| Clio `GET /internal/v1/observations/status` | Collection progress, errors and source freshness |
| Clio `GET /internal/v1/observations/monitoring` | Worker and data monitoring |
| Prophet `GET /internal/v1/forecasts/{product}/status` | Current release freshness and latest generation attempt |

These differ from `/health/live` (HTTP process) and `/health/ready` (storage
readiness). Docker Compose also uses `clio check-health worker` for worker
heartbeats. Diagnostic reads do not trigger collection or generation.

Inspect Prophet's `forecast_run`, `forecast_artifact`, `forecast_release` and
`forecast_slot` tables directly for run evidence and schedule history. The CLI
keeps `verification-report` because it computes metrics over saved pairs rather
than just listing database rows.

## Intelligence

| Command | Purpose |
| --- | --- |
| `./argus intelligence process [product]` | Process available forecasts once, currently in stub mode |
| `./argus intelligence status [product]` | Show processing status |

## Users

```bash
./argus api user <action> [arguments]
./argus api user --help
```

## Logs

```bash
./argus logs [service] [--tail N] [-f]
```

Shows the last 100 lines by default; `-f` follows new output.

Reads Docker Compose logs in both environments. `clio` includes `clio` and
`clio-worker`; `prophet` includes its worker and HTTP service. Other local services:
`api`, `intelligence`, `intelligence-api`, `postgres`, `redis`; `clio-worker` and `prophet-api` can also
be selected individually. Production additionally supports `frontend`, `nginx`
and `alloy`. Without a service, shows all Compose logs.

## Historical recovery

Restore missing historical observations using each configured depth, or a UTC
range with an exclusive end (maximum 31 days for explicit ranges):

```bash
./argus clio backfill
./argus clio backfill kp dst f10_7
./argus clio collect sdo
./argus clio backfill sdo
./argus clio collect gong goes
./argus clio backfill goes                   # source samples; calibration runs in Prophet
./argus clio backfill v n t --from 2026-08-31 --to 2026-09-09
```

Both `--from` and `--to` must be supplied together. Dates mean midnight UTC;
timestamps must include a timezone and fall on a whole UTC hour. The end must
not be in the future. Without these flags, each metric uses its configured
`backfill.days`.

AIA/HMI are included by default when enabled; select `sdo` to collect only images.
Their backfill is limited to the retained 45 days, including when an explicit range
is supplied. Expired image cleanup runs automatically in the worker.

Existing measurements are preserved. The output reports source failures and
remaining raw slots at each metric's cadence (hourly, three-hourly or daily).
Only closed, fully contained slots are checked. The worker runs
both `clio.observations.<metric>.schedules.live` and `schedules.backfill`
automatically. See the Clio README for archive coverage limits.
The shared SDO collector handles all AIA channels; no separate AIA193 collector remains.
Clio restores missing AIA/GONG originals only when their bytes match the recorded
SHA256. Prophet rebuilds its own derivative caches from these originals. Existing observations
and first receipt timestamps are preserved.
