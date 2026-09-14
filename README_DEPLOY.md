# Deployment

A `v*` tag starts [.github/workflows/deploy.yml](.github/workflows/deploy.yml).
The local `./deploy.sh [patch|minor|major] [-m message]` helper uploads models and
metrics, commits/pushes notebooks, the private backend and the main repository,
then creates the next version tag. Review these checkouts before using it.
Model/metric uploads remain separate from application deployment.

## Builds in GitHub Actions

Actions fingerprints each image's source files, Dockerfile and lockfiles. Images
with matching inputs are reused from the registry; only missing identities are
built, with BuildKit caching. This avoids depending on the previous tag: skipped
or failed releases do not hide source changes. The private backend is resolved
once, and Clio/Prophet builds check out that exact commit.

| Change | Images affected |
| --- | --- |
| Frontend source/manifests | Frontend |
| API source/dependencies/migrations | API |
| Clio source/dependencies/migrations | Clio |
| Prophet source/dependencies/migrations | Prophet |
| Intelligence source/dependencies | Intelligence |
| `packages/common` | API, Clio, Prophet, Intelligence |
| `packages/forecast` | API, Prophet |
| `packages/clio`, private backend | Clio, Prophet |
| Top-level docs and deployment configuration | None |

The COPY input map lives in `scripts/deployment/release.py`, which runs **only in
Actions**. A test checks it against the Dockerfiles. Matrix jobs still start to
check the registry, but unchanged images are not rebuilt. Package documentation
copied into an image participates conservatively in its identity.

All five image references are pinned by digest. Use **Run workflow → rebuild=true**
to refresh application base images explicitly. Infrastructure images are downloaded
only when missing; upgrading those is a separate explicit maintenance operation.

## Applying on the server

Actions uploads a release directory and runs `bash <release>/deploy.sh /var/www`
over SSH. The host needs **Bash, Docker Compose v2, rsync and flock**. No host
Python, extra deployment container or service is required. Compose must support
`pull --policy missing` and `up --wait`.

Existing GitHub secrets and server `.env` / `.env.local` remain required.
**No new manual environment settings or secrets.** Actions generates the five
`ARGUS_*_IMAGE` values in `.release-images.env`; use the wrapper below so these
pinned versions are always included. Do not define competing image overrides.

The short server script:

1. Validates Compose and downloads missing images before stopping services.
2. Compares migration/config fingerprints with the last **successful** deployment.
   Stops writers of domains with changed migrations and backs up PostgreSQL.
3. Synchronizes repository-owned `configs`, `nginx` and `alloy` directories;
   operator env files, data and models are outside these directories.
4. Applies changed domain migrations and runs `docker compose up -d --wait`.
   Compose recreates containers whose images/configuration changed; unchanged
   containers remain running. Mounted common config changes stop/restart Python
   consumers; nginx/alloy config changes restart their services.
5. Reloads nginx to refresh upstream DNS and records successful fingerprints.

A docs-only release does not rebuild or recreate application containers. It still
runs the Compose reconciliation and nginx validation/reload. There is no separate
service-state planner, forced recreation of all services, routine bootstrap,
crontab replacement or image pruning.

The first release builds missing fingerprinted images and runs all existing domain
migrations once. PostgreSQL must already be provisioned. Historical Alembic files
remain necessary; provisioning/password maintenance is described in
[domain storage](docs/domain-storage.md).

## Commands and failures

Deployment installs a wrapper that works from any directory:

```bash
/var/www/bin/argus prophet status solar-wind-speed
/var/www/bin/argus prophet slots --limit 10
/var/www/bin/argus clio refresh
/var/www/bin/argus compose ps
```

Add `/var/www/bin` to PATH to use `argus` directly. Domain commands acquire a shared
host lock, preventing overlap with deployment; competing domain writers also use
their existing database locks. Direct `argus compose` maintenance is an operator
escape hatch: do not run it concurrently with deployment.

A failure stops the script and leaves successful fingerprints unchanged. Affected
writers can remain stopped; fix the error and rerun the release. Migrations are
retried, and Compose reconciles actual containers. The uploaded bundle and
pre-migration database backup remain available. There is no automatic database
rollback; a restore requires accounting for other domains' subsequent writes.

Actions serializes production deployments, and flock prevents concurrent host
scripts. Production rollout and real image builds must still be verified by the
workflow; local orchestration tests use a fake Docker executable.

## CI timing

Frontend exports `mode=min` BuildKit cache; Python images retain `mode=max`.
This reduces frontend cache export volume, but intermediate dependency/build
layers will no longer be exported and may need rebuilding on fresh runners.
Compare total build duration across subsequent releases, not export time alone.
The server script logs elapsed seconds for image downloads, writer shutdown,
backup, configuration, migrations, application readiness and reloads. These
messages distinguish remote execution time from SSH/SCP action overhead.

Intelligence is an explicit tools-profile job: `argus intelligence check` or
`argus intelligence check dst --release-id <uuid>`. Its image is pulled during
deployment but no background container or schedule starts. It shares only the
existing forecast HTTP token, has no database credentials or mounted data, and
adds no manual env requirements. See [Intelligence](apps/intelligence/README.md).
