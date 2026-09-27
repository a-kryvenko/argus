# Command reference

Two supported forecast workflows, their protocols and MLflow/site publication:
[Forecast workflows](forecast-workflows.md).

```bash
./argus observe solar-wind-speed
./argus prophet verify solar-wind-speed
./argus prophet verification-report solar-wind-speed
```

Model evaluation, MLflow export and optional website metrics saving are separate
cells in `notebooks/evaluate_models.ipynb`; there is no model evaluation CLI.

Run `./argus` from the project root locally or `/var/www` in production.
Command arguments are identical in both environments. Python operations are exposed
through `./argus`; the former pnpm `app:*` / `db:*` aliases have been removed.
Service CLIs validate their own arguments; shell adapters handle Compose,
migrations and cross-service workflows.

Use `./argus clio --help` or `./argus prophet --help` for service commands.
Clio handlers share one event-loop/connection cleanup boundary.
`clio refresh` collects both live sources and normalizes observations;
`clio refresh observations` only runs the model-observation refresh. Scheduled
refresh retains the latter behavior.

## Help

```bash
./argus help [command]
```

Shows available commands and the selected environment.

## Service processes

`./argus compose <arguments>` runs Docker Compose for the selected environment,
loading its env files. In dev:

```bash
./argus compose up -d --wait
pnpm dev                         # frontend, in this terminal
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

## Refresh and status

`refresh` runs once and exits. Omitting the source or product selects `all`.
Clio execution stops on the first error. Prophet attempts every selected product
and publishes successes independently; it exits with an error if any product fails.
Omitting the product for `status` also selects `all`. Multi-product status and
Intelligence refresh return a JSON array from one service invocation.

| Command | Purpose |
| --- | --- |
| `./argus clio refresh [source]` | Collect or update observations |
| `./argus clio aggregate [--limit N]` | Process queued aggregates |
| `./argus clio status` | Show collection progress and source freshness |
| `./argus clio worker` | Run both collectors and scheduled normalization/aggregation in the foreground |
| `./argus prophet refresh [product]` | Generate and publish forecasts |
| `./argus prophet status [product]` | Show the current release and latest generation attempt |
| `./argus intelligence refresh [product]` | Process available forecasts once, currently in stub mode |
| `./argus intelligence status [product]` | Show processing status |

Clio sources: `observations` (normalized observations and input history),
`solar-wind`, `geomagnetic`, `all`. With `all`, both collectors run first,
followed by `observations`.

Clio manual commands use the same locks as the worker and do not advance its
scheduled completion markers. They also work when the worker is stopped.
Docker Compose starts the Clio worker in both dev and production.

Products: `solar-wind-speed`, `solar-wind-density`, `geomagnetic-activity`, `dst`,
`hmf`, `atmospheric-density`, `all`. Prophet refresh calculates only the selected
product; `all` calculates every supported product using one observation snapshot.
`solar-radiation` is not supported by these commands.

## Users

```bash
./argus api user <action> [arguments]
./argus api user --help
```

Manage dashboard users; use `--help` for available actions and arguments.

## Logs

```bash
./argus logs [service] [--tail N] [-f]
```

Shows the last 100 lines by default; `-f` follows new output.

Reads Docker Compose logs in both environments. `clio` includes `clio` and
`clio-worker`; `prophet` includes its worker and HTTP service. Other local services:
`api`, `intelligence`, `postgres`, `redis`; `clio-worker` and `prophet-api` can also
be selected individually. Production additionally supports `frontend`, `nginx`
and `alloy`. Without a service, shows all Compose logs.

Restore missing historical observations (UTC, exclusive end, maximum 31 days):

```bash
./argus clio backfill --from 2026-08-31 --to 2026-09-09
```

Existing measurements are preserved. The output reports source failures and
remaining raw observation gaps; see the Clio README for archive coverage limits.
