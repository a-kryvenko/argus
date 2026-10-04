# Development and deployment

## Local development

Requires Docker Compose v2, Node.js 20.9+, pnpm 9, and private
`packages/forecast-core` / `packages/intelligence-core` checkouts.
Forecasting also needs the models and inputs referenced by `configs/`.

```bash
pnpm install --frozen-lockfile
mkdir -p data
```

Configure `.env` and `.env.local` (local values take precedence):

| Variables | Purpose |
| --- | --- |
| `DB_HOST/PORT/NAME/USER/PASSWORD` | Database administrator; provisioning only |
| `API_DB_*`, `CLIO_DB_*`, `PROPHET_DB_*`, `INTELLIGENCE_DB_*` | Separate database credentials per service: `HOST`, `PORT`, `NAME`, `USER`, `PASSWORD` |
| `OBSERVATIONS_SERVICE_TOKEN` | Shared by Clio, API and Prophet |
| `FORECASTS_SERVICE_TOKEN` | Shared by Prophet, API and Intelligence |
| `INTELLIGENCE_SERVICE_TOKEN` | Shared by API and Intelligence; defaults to `FORECASTS_SERVICE_TOKEN` in Compose |
| `SDO_EMAIL` | Email for SDO data requests |
| `DASHBOARD_ORIGINS`, `DASHBOARD_COOKIE_SECURE` | `http://localhost:3000`, `false` for local HTTP |

Use raw passwords; single-quote values containing `$`. Dev Compose sets database
connections to `postgres:5432` and internal service URLs automatically.
Host tools use `127.0.0.1:5432`.

```bash
# First startup
./argus compose build
./argus compose up -d --wait postgres redis
./argus db provision --apply
./argus db migrate
./argus compose up -d --wait
pnpm dev
```

Frontend: `localhost:3000`. API / Clio / Prophet: `localhost:8000` / `8001` / `8002`.
`pnpm dev` runs the frontend; Compose runs the backend.

```bash
# Initial data and forecasts, in another terminal
./argus clio refresh
./argus clio aggregate
./argus prophet refresh
./argus intelligence refresh

# Daily use
./argus compose up -d --wait
./argus compose restart clio-worker prophet intelligence # After worker code changes
./argus logs clio -f
./argus compose down
```

HTTP services reload Python changes. Restart workers and services affected by
configuration changes; rebuild images after dependency or Dockerfile changes.
`compose down --volumes` also deletes local database and Redis volumes.

## Production setup

Server requirements: Bash, Docker Compose v2, rsync, flock, and operator-managed
`.env` / `.env.local` in `/var/www`. Compose must support `pull --policy missing`
and `up --wait`.

- Configure all four service databases with distinct owners, usually at `postgres:5432`.
  Preserve existing PostgreSQL credentials and volumes.
- Set service tokens and the public dashboard origin; use secure cookies for HTTPS.
- Dashboard login has two limits: nginx allows 10 immediate attempts per client IP,
  then replenishes one attempt every 10 seconds; the API allows 10 attempts per
  username per 15-minute window across all clients. nginx resolves client IPs only
  through the trusted edge network (`172.20.0.0/16`). Keep the API internal; direct
  local development requests only have the username limit. Apply the nginx and API
  changes together when deploying this configuration.
- Add SSH and private-checkout secrets required by [deploy.yml](.github/workflows/deploy.yml).
- Add `GHCR_USERNAME` and `GHCR_READ_TOKEN`: a classic PAT with `read:packages`
  and access to all five images; authorize organization SSO if required.
- Install the release bundle and Compose/image configuration before initial provisioning.
  Start PostgreSQL/Redis, then run `./argus db provision --apply` and
  `./argus db migrate` before starting applications. Normal deployments do not provision databases.

Provisioning only creates missing databases/owners; it does not rotate passwords.
Models and metrics are uploaded separately from application images.

## Release

Install test dependencies once:

```bash
pnpm install --frozen-lockfile
pnpm --filter web exec playwright install chromium
```

Run web checks separately with `pnpm --filter web test` (unit tests) and
`pnpm --filter web test:dashboard` (browser tests). Browser tests start an isolated
Turbopack dev server with its cache in `.next/playwright`. Failure screenshots
are always enabled; set `PLAYWRIGHT_SCREENSHOTS=1` to also capture the manual
review screenshots from successful tests.

Review the main, notebooks and private-backend checkouts before publishing:

```bash
./deploy.sh                         # Checks only
./deploy.sh -m "message"             # Checks, commits and pushes; no release
./deploy.sh patch -m "message"       # Also uploads artifacts and pushes a release tag
# Use minor or major instead of patch when needed; a bump requires -m.
```

All modes run web lint, TypeScript, Node, Playwright and Python tests before
publishing. Python tests require `uv`; run them separately with `./scripts/test-python -q`.

The login edge regression checks use a disposable nginx container and mock HTTP
upstreams, and run in PR CI. To run them locally (requires Docker):

```bash
TEST_NGINX_IMAGE=nginx:stable-alpine tests/runtime/.venv/bin/python -m pytest -q tests/integration/test_login_edge.py
```

A `v*` tag triggers GitHub Actions. It builds or reuses images, pins their digests
in `.release-images.env`, uploads the release bundle, and deploys to `/var/www`:

1. Lock deployment, validate configuration and pull images.
2. Stop affected writers and back up PostgreSQL before migrations (`/var/www/backups/`).
3. Install release files, apply changed migrations and reconcile services.
4. Reload nginx and record the successful release fingerprints.

Operator env, models and data remain separate. Do not override generated image references.
`./argus` defaults to dev mode; deployment sets `.argus-mode` to `prod`.

## Maintenance and recovery

Use `/var/www/argus` for production commands and logs; see the [command reference](docs/commands.md).
Before manual schema changes, stop affected writers and back up databases:
the manual CLI locks operations but does neither automatically.
Password changes require updated env and container recreation.

If deployment fails, inspect the failing phase, fix it and rerun the same release.
Affected writers may remain stopped. There is no automatic database rollback;
restoring a full-instance backup affects every domain and must be coordinated with
application images, models and writes made since the backup.

## Reference

- [Commands](docs/commands.md) · [Architecture and database ownership](docs/architecture.md)
- [Clio](apps/clio/README.md) · [Prophet](apps/prophet/README.md) · [Intelligence](apps/intelligence/README.md) · [API](apps/api/README.md)
