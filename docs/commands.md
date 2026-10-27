# Commands

`./argus help` and `./argus <service> --help` list options.
The wrapper selects dev/prod Compose and loads env files.
[First startup](../README_DEPLOY.md#local-development).

## Service processes

```bash
./argus compose up -d --wait
pnpm dev                         # Local frontend
./argus compose restart prophet # Worker source changes
./argus compose down
```

## Databases

Services: `api`, `clio`, `prophet`, `intelligence`.

| Command | Purpose |
| --- | --- |
| `./argus db provision [--apply]` | Preview/create owners and databases |
| `./argus db migrate` | Upgrade all services; stop on first error |
| `./argus <service> migrate` | Upgrade one service to head |
| `./argus <service> migration create -m "description"` | Create a migration; dev only |
| `./argus <service> migration current` / `history` | Inspect revisions |
| `./argus <service> migration downgrade <revision>` | Downgrade; may delete data |

API/Clio creation supports `--autogenerate`; Prophet/Intelligence use handwritten migrations.
Manual migrations do not stop writers or create backups.

## Clio

| Command | Purpose |
| --- | --- |
| `./argus clio serve` | Run the observation API |
| `./argus clio collect [METRIC ...]` | Collect live observations; default: all |
| `./argus clio backfill` | Check and fill configured historical horizons |
| `./argus clio normalize` | Write hourly rows from live observations |
| `./argus clio cleanup` | Remove expired live observations, hourly rows and SDO files |

Backfill takes no arguments; horizons come from `clio.observations.*.backfill_days`.
Cleanup uses `clio.retention.live_days` and `normalized_days` (6 and 60).
Restored originals must match their recorded SHA256.
Partial results and reported failures exit with code 1.
Cron: `apps/clio/cronjobs.txt`. Logs: `log/clio/`.

## Prophet

| Command | Purpose |
| --- | --- |
| `./argus prophet generate [product]` | Generate/publish; `refresh` alias; default all |
| `./argus prophet verify [product] [--days 7]` | Verify saved releases against observations |
| `./argus prophet verification-report [product] [--days 7]` | Scores by artifact/model hash |
| `./argus prophet cleanup [--days 90] [--apply]` | Preview/delete eligible history |

Generation products: [catalog](../apps/prophet/README.md#products).
Successes publish independently; any selected failure produces a nonzero exit.
Verify/report exclude atmospheric density, default to speed and accept 1–25 days.
Cleanup preserves current releases and latest attempts.

## Intelligence

| Command | Purpose |
| --- | --- |
| `./argus intelligence worker` | Poll enabled density profiles and forecast integration |
| `./argus intelligence process [product]` | One-shot forecast integration check; stub results |
| `./argus intelligence status [product]` | Integration status |

Risk assessment uses [density-advisory HTTP routes](density-advisory.md#service-contract).

## Diagnostics

| Route | Purpose |
| --- | --- |
| Clio `/internal/v1/observations/status` | Collection errors/freshness |
| Clio `/internal/v1/observations/monitoring` | Worker/data monitoring |
| Prophet `/internal/v1/forecasts/{product}/status` | Release freshness and latest attempt |
| `/health/live`, `/health/ready` | HTTP process/local storage |

Internal data routes require service tokens. Reads do not collect or generate.

## Users

```bash
./argus api user --help
```

## Logs

```bash
./argus logs [service] [--tail N] [-f]
```

Default: last 100 lines; no service selects all. `clio` selects the HTTP service;
`prophet` includes HTTP. Processes can also be selected individually.
Production adds `frontend`, `nginx` and `alloy`.

[Historical demo](demo-mode.md) · [Deployment](../README_DEPLOY.md)
