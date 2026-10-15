# Development and deployment

## Local development

Requires Docker Compose v2, Node.js 20.9+, pnpm 9, private `forecast-core` and
`intelligence-core` checkouts, and model artifacts referenced by `configs/`.

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

```bash
pnpm install --frozen-lockfile
mkdir -p data
./argus compose build
./argus compose up -d --wait postgres redis
./argus db provision --apply
./argus db migrate
./argus compose up -d --wait
pnpm dev
```

Frontend/API/Clio/Prophet: localhost ports 3000/8000/8001/8002.
Workers collect and generate automatically. [Manual commands](docs/commands.md).

| Change | Apply with |
| --- | --- |
| API, Clio or Prophet HTTP source | Automatic dev reload |
| Worker or Intelligence source | `./argus compose restart <service>` |
| Env, mounts or Compose | `./argus compose up -d <service>` |
| Dependencies or Dockerfile | `./argus compose up -d --build <service>` |

Data mounts follow [service ownership](docs/architecture.md#runtime-ownership).
Prophet demo caches use `data/demo`; legacy `data/raw` needs an explicit one-off mount.
`compose down --volumes` deletes local DB and Redis data.

## Production setup

Server: Bash, Docker Compose v2, rsync, flock; operator env files in `/var/www`.
Compose must support `pull --policy missing` and `up --wait`.

- Configure distinct DB owners, three service tokens, HTTPS dashboard origin and secure cookies.
- Preserve existing PostgreSQL credentials/volumes. Provisioning creates missing owners/databases; it does not rotate passwords.
- Configure SSH/private-checkout secrets from [deploy.yml](.github/workflows/deploy.yml), plus `GHCR_USERNAME` and `GHCR_READ_TOKEN` (`read:packages`, access to all five images).
- Install release files, start PostgreSQL/Redis, provision and migrate before starting applications. Normal releases do not provision.
- Upload configured model artifacts and static metrics separately. A missing/corrupt model bundle is a deployment error.

Keep API internal. Login limits: nginx 10 immediate attempts/IP, then one per
10 seconds; API 10 attempts/username per 15 minutes. nginx trusts client IPs only
from `172.20.0.0/16`; deploy edge and API changes together.

## Release

```bash
pnpm --filter web exec playwright install chromium
./deploy.sh                         # Checks and diagram generation
./deploy.sh -m "message"             # Also commits and pushes; no release
./deploy.sh patch -m "message"       # Also uploads artifacts and tags a release
```

`minor`/`major` also require `-m`. Checks include web lint/types, Node, Playwright
and Python (`uv` required). Python: `./scripts/test-python -q`; diagrams require
Python 3.10+, Java 21+ and curl. [Diagram commands](docs/architecture/README.md).

Browser tests use `.next/playwright`; failures save screenshots.
`PLAYWRIGHT_SCREENSHOTS=1` also saves successful review screenshots.
Optional nginx checks require Docker and `TEST_NGINX_IMAGE=nginx:stable-alpine`.

A `v*` tag builds/reuses images and deploys their pinned digests:

1. Lock deployment, validate config, prune unreferenced images older than seven days.
2. Stop affected writers; back up PostgreSQL under `/var/www/backups/` before migrations.
3. Install files, migrate, reconcile services and reload nginx.
4. Record fingerprints; after a successful backup/deployment, retain seven timestamped dumps.

Failed backups remain `.partial`; failed deployments do not rotate backups.
Operator env/models/data remain separate; do not override generated image references.
`./argus` defaults to dev; production sets `.argus-mode` to `prod`.

## Maintenance and recovery

Use `/var/www/argus` in production. Manual schema commands lock operations but do
not stop writers or back up DBs; do both first. Credential changes require container recreation.

After a failed deployment, fix the failing phase and rerun the same release;
writers may remain stopped. DB rollback is manual. Full-instance restore affects
every domain and must match images, models and subsequent writes.

[Commands](docs/commands.md) · [Architecture](docs/architecture.md)
