# Development and deployment

## Local development

Requires Docker Compose v2, Node.js 20.9+, pnpm 9, private `forecast-core` and
`intelligence-core`.

Configure `.env` / `.env.local` (local values override):

| Variables | Purpose |
| --- | --- |
| `DB_HOST/PORT/NAME/USER/PASSWORD` | Administrator; provisioning only |
| `API_DB_*`, `CLIO_DB_*`, `PROPHET_DB_*`, `INTELLIGENCE_DB_*` | Separate service owners: `HOST`, `PORT`, `NAME`, `USER`, `PASSWORD` |
| `OBSERVATIONS_SERVICE_TOKEN` | Clio, Prophet and API |
| `FORECASTS_SERVICE_TOKEN` | Prophet, Intelligence and API |
| `INTELLIGENCE_SERVICE_TOKEN` | Intelligence and API |
| `SDO_EMAIL` | SDO requests |
| `DASHBOARD_ORIGINS`, `DASHBOARD_COOKIE_SECURE` | Local: `http://localhost:3000`, `false` |

Use three distinct service tokens. Passwords are raw; single-quote values containing `$`.
Compose sets internal URLs and DB hosts; host tools use `127.0.0.1:5432`.
Service SQL uses unqualified table names with `<service>, public` in `search_path`.
Tables and Alembic version records remain in the existing service schemas until the next migration.

```bash
pnpm install --frozen-lockfile
mkdir -p data
./argus compose build
./argus compose up -d --wait postgres redis storage
./argus db provision --apply
./argus db migrate
./argus compose up -d --wait
pnpm dev
```

Frontend/API/Clio/Prophet: localhost ports 3000/8000/8001/8002.
Dev uses SeaweedFS (`storage`, persistent `seaweed-data` volume), with automatic
`argus-private`/`argus-public` buckets and separate writer/reader keys. Public files:
`http://localhost:8333/argus-public`; override with `DEV_S3_PORT` and `DEV_S3_PUBLIC_BASE_URL`.
Services use S3 directly in both environments. Production uses `S3_ENDPOINT_URL`,
`S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` and optional `S3_REGION` (default `auto`).
Clio stores file observations in S3: private SDO under `argus-private/images/SDO`,
public AIA under `argus-public/images/AIA`, other sources under `argus-private/observations`.
`clio.files` stores object references; `clio.file_observation` stores their context. Catalogs read metadata from PostgreSQL; file content stays in S3.
Production `S3_PUBLIC_BASE_URL` defaults to `https://media.argussun.com`. Prophet commands use
`S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY` with read/write access; charts use `argus-public/forecasts/summary/`.
Prophet HTTP/PROSWIN use `S3_READ_ACCESS_KEY_ID`/`S3_READ_SECRET_ACCESS_KEY` for private sources.
Prophet HTTP reads the prepared summary from the shared `data/prophet` directory (read-only mount).
Add `configs/s3-summary-lifecycle.json` to the public bucket lifecycle (SVG expiry: two days).

One-time local archive migration (after migrating Clio):
```bash
./argus compose run --rm --no-deps -v ./data/observations:/legacy/observations:ro clio-worker .venv/bin/python -m clio.migrate_local_files --root /legacy/observations
./argus compose run --rm --no-deps -v ./data/observations:/legacy/observations:ro clio-worker .venv/bin/python -m clio.migrate_local_files --root /legacy/observations --apply
```
Scientific SDO migration selects only AIA171/AIA211 in the latest 96 hourly slots; other NPZs are counted as `skipped`.
Both commands accept `--kind` and `--limit`. Errors produce exit code 1; reruns preserve uploaded objects. Local files are retained.

Hourly cleanup retains scientific SDO for 96 hours, latest public AIA per channel, GONG/GOES for their configured backfill windows, and DONKI/ENLIL for `clio.solar_wind_events.retention_days` (30 by default).
DONKI stores unchanged polls without duplicate payloads; cleanup also compacts existing duplicates. Unreferenced objects older than 15 minutes are deleted in batches of up to 1000 per source; failed deletions are retried.

Workers collect and generate automatically. [Manual commands](docs/commands.md).

Data mounts follow [service ownership](docs/architecture.md#runtime-ownership).
Prophet demo caches use `data/demo`;

## Production setup

Server: Bash, Python 3.10+, Docker Compose v2, rsync, flock; operator env files in `/var/www`.
Compose must support `pull --policy missing` and `up --wait`.

- Configure distinct DB owners, three service tokens, HTTPS dashboard origin and secure cookies.
- Preserve existing PostgreSQL credentials/volumes. Provisioning creates missing owners/databases; it does not rotate passwords.
- Configure SSH/private-checkout secrets from [deploy.yml](.github/workflows/deploy.yml), plus `GHCR_USERNAME` and `GHCR_READ_TOKEN` (`read:packages`, access to all six images).
- Install release files, start PostgreSQL/Redis, provision and migrate before starting applications. Normal releases do not provision.
- Create a smoke account with `/var/www/argus api user create deploy-smoke` (no groups). Set `DEPLOY_SMOKE_USERNAME` and `DEPLOY_SMOKE_PASSWORD` in `.env.local`.
- `DEPLOY_MIN_AVAILABLE_MIB` defaults to 1024; each HTTP replacement requires this much available host RAM.

Keep API internal. Login limits: nginx 10 immediate attempts/IP, then one per
10 seconds; API 10 attempts/username per 15 minutes. nginx trusts client IPs only
from `172.20.0.0/16`; deploy edge and API changes together.

## Release

```bash
./deploy.sh                         # Checks and diagram generation
./deploy.sh -m "message"             # Also commits and pushes; no release
./deploy.sh patch|minor|major -m "message"       # Also uploads artifacts and tags a release
```

`minor`/`major` also require `-m`. Checks include web lint/types, Node, Playwright
and Python (`uv` required).

Diagrams require Python 3.10+, Java 21+ and curl. [Diagram commands](docs/architecture/README.md).

A `v*` tag deploys pinned images. Releases require existing `.rolling-state.json`
and replace changed HTTP services individually: readiness, smoke checks, graceful routing switch, request
drain, old-container stop. Workers remain single-instance and update separately.
One `internal-router` nginx serves Clio/Prophet/Intelligence/API/frontend on ports
8101–8105, bound to their service networks. The first release migrates and removes the old per-service routers.

Smoke checks cover HTML/static assets, login/logout, observations, forecasts and
Intelligence reads. No inference runs. Models/metrics use verified artifact bundles;
configs are versioned. Demo data remains worker-owned; old frontend assets are retained.

Rolling deployment requires `online_compatible = True` in every pending migration,
reviewed for compatibility with both application versions. Missing or false declarations
require a maintenance deployment. Actual DB revisions are checked before
backup; changed files alone do not trigger one. Backups run before stopping services,
with fast compression and progress every 15s. Migrations use a 5s lock timeout and
5min statement timeout; affected workers drain while HTTP stays online.
Current migration histories remain intact.

Deployment waits up to 600s for running jobs to release `.deployment.lock` (`DEPLOY_LOCK_TIMEOUT`); timeout exits with status 75.

Failed backups remain `.partial`; failed deployments do not rotate backups.
Retain three pre-migration backups; prune unused images older than 72 hours.
Operator env/models/data remain separate; do not override generated image references.
Production CLI requires `.rolling-state.json`, `.rolling-compose.json` and `.release-compose.json`.
`./argus` defaults to dev; production sets `.argus-mode` to `prod`.

## Maintenance and recovery

Use `/var/www/argus` in production. Manual schema commands lock operations but do
not stop writers or back up DBs; do both first. Credential changes require container recreation.

Failed HTTP replacements restore the previous route; completed replacements remain
active. Rerun the same release to recover an interrupted switch and finish deployment.
Use `argus` for production commands/logs; it selects active HTTP slots. Bare
`docker compose up` bypasses this state. DB rollback is manual. Preserve `releases/`,
`artifacts/`, `.rolling-*` and stopped previous slots for recovery.

[Commands](docs/commands.md) · [Architecture](docs/architecture.md)

Clio runs in `clio-worker`; logs: `./argus logs clio`.
Deployment replaces host cron with `apps/{prophet,api,intelligence}/cronjobs.txt` and `.deploy/cronjobs.txt`; `@ROOT@` expands to the deployment directory.
