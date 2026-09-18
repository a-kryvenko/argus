# Command reference

Run `./argus` from the project root locally or `/var/www` in production.
Command arguments are identical in both environments.

## Help

```bash
./argus help [command]
```

Shows available commands and the selected environment.

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
Omitting the product for `status` also selects `all`.

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
`pnpm dev` includes the Clio worker; production Compose starts it automatically.

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

Locally, reads `data/logs/<service>.log` for `api`, `clio`, `prophet` or
`intelligence`. Without a service, reads all available local log files.

In production, reads Docker Compose logs. `clio` includes `clio` and `clio-worker`;
`prophet` includes its worker and HTTP service. Other supported
services: `api`, `intelligence`, `frontend`, `nginx`, `postgres`, `redis`, `alloy`.
Without a service, shows all Compose logs.
