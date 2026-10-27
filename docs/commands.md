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
| `./argus clio collect [METRIC ...]` | Collect live observations |
| `./argus clio backfill [METRIC ...] [--from DATE --to DATE]` | Fill historical gaps |
| `./argus clio normalize` | Normalize stored observations |
| `./argus clio sdo-cleanup` | Prune expired SDO files |

One-shot commands exit after completion. Metrics are space-separated (`v n t`);
omitting them selects all configured observations. Empty DBs need collection
before normalization. Deployment installs `apps/clio/cronjobs.txt`; flock prevents overlap.
Logs are in `log/clio/`.
Use `backfill sdo --batch` for bounded SDO warmup.

## Historical recovery

```bash
./argus clio backfill
./argus clio backfill v n t --from 2026-08-31 --to 2026-09-09
./argus clio backfill sdo
./argus clio backfill gong goes
```

Supply both dates; end is exclusive, not future, range ≤31 days. Dates mean UTC
midnight; timestamps require a timezone and whole UTC hour. Without dates, use
configured `backfill.days`. SDO is limited to 45-day retention.

Existing values/first receipts remain unchanged; restored files must match saved
SHA256. Prophet rebuilds derivative caches from originals. Output reports source
failures and missing closed slots at each metric's cadence.

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
