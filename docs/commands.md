# Commands

`./argus help` and `./argus <service> --help` list options.
The wrapper selects dev/prod Compose and loads env files.
Shared services: `docker-compose.yml`; overrides: `docker-compose.override.yml` (dev), `.deploy/docker-compose.yml` (prod).
[First startup](../README_DEPLOY.md#local-development).

## Service processes

```bash
./argus compose up -d --wait
pnpm dev                         # Local frontend
./argus compose restart clio-worker # Scheduler source changes
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

Backfill fills missing native slots within `clio.observations.*.backfill_days`.
Published means need only a recent release.
Cleanup uses `clio.retention`; restored originals must match their SHA256.
Scheduler: `clio-worker` in dev and prod. Logs: `./argus logs clio`.

## Prophet

| Command | Purpose |
| --- | --- |
| `./argus prophet serve [--host HOST] [--port 8000]` | Serve published forecasts |
| `./argus prophet generation [product]` | Generate/publish; default all |
| `./argus prophet summary` | Prepare local summary and upload SVGs to S3 |
| `./argus prophet verification [product]` | Verify saved releases against observations |
| `./argus prophet cleanup [--apply]` | Preview/clear diagnostics older than 14 days; preserve forecasts |

Generation products: [catalog](../apps/prophet/README.md#products).
Successes publish independently; any selected failure produces a nonzero exit.
Verification defaults to speed; window: `prophet.verification_days` (30 days).
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
| Clio `/internal/v1/observations/status` | Stored observation availability and freshness |
| Prophet `/internal/v1/forecasts/status` | All products: release freshness and latest attempt |
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
