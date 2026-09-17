# Deployment

## Prerequisites

The server needs Bash, Docker Compose v2, rsync and flock, plus operator-owned
`.env` / `.env.local` in `/var/www`. Compose must support `pull --policy missing`
and `up --wait`. Models and metrics are uploaded separately from application images.

Configure each service's `<DOMAIN>_DB_HOST/PORT/NAME/USER/PASSWORD`; production host
is normally `postgres`, port `5432`. Explicit env values take precedence, so do not
copy local `127.0.0.1` settings to the server. Passwords are raw strings. Administrator
`DB_*` credentials are reserved for PostgreSQL and explicit provisioning.
Keep existing PostgreSQL initialization credentials and persistent volumes intact.

Configure shared service tokens and public origins as described in
[setup](README.md#local-development). GitHub Actions requires the SSH/registry/private
checkout secrets referenced by [deploy.yml](.github/workflows/deploy.yml).
It generates `.release-images.env` with five digest-pinned image references;
do not override them with competing image settings.

## Private application images (GHCR)

Application images are published to `ghcr.io/a-kryvenko/argus-*`. Actions publishes
with its automatic `GITHUB_TOKEN` and job-level `packages: write` permission;
no personal write token is needed. Images are linked to this repository.
New GHCR packages are private by default. If these package names already exist,
verify that each package has **Private** visibility and grants this repository
Actions access before running the release. Repository visibility and package
visibility are separate settings.

Before the first GHCR deployment, add these repository Actions secrets:

- `GHCR_USERNAME`: GitHub username of the account allowed to read all five packages.
- `GHCR_READ_TOKEN`: that account's personal access token **(classic)** with
  `read:packages`. Authorize SSO if required by the organization. Fine-grained
  personal access tokens are not supported for this registry authentication.

The deployment logs the production SSH user into `ghcr.io` with this read token
via standard input. Docker stores the credentials for subsequent manual pulls;
use the same server user for maintenance and update the secret when rotating the
token. The existing production SSH and private-backend checkout secrets remain
required.

The first release builds any images absent from GHCR and installs digest-pinned
references. Existing Docker Hub images are not copied or deleted. Keep the old
images and server Docker Hub login while historical release bundles are needed
for recovery. After a successful migration, the `DOCKERHUB_LOGIN` and
`DOCKERHUB_TOKEN` Actions secrets are no longer used and can be removed.
Infrastructure and Dockerfile base images still use their upstream registries.

See [GitHub's Container registry documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
for authentication, package visibility and access settings.

## Initial databases and maintenance

With the release bundle, root `argus`, prod adapters and Compose/image configuration
installed, start PostgreSQL/Redis, then run `./argus db provision --apply` and
`./argus db migrate` before starting applications. All four domain databases must
exist before normal deployment, which applies migrations but never provisions.

Provisioning creates only missing databases/owners and never rotates passwords.
Existing roles require their current passwords. Foreign ownership or privileged
service roles are rejected. Database creation is not transactional: correct the
error and repeat after a partial failure. Alembic creates the service schemas.

Before manual schema changes, stop affected writers and back up the databases.
The manual CLI takes an exclusive host lock but does not stop containers or make
a backup. Password rotation is separate: stop affected services, change the owner's
password, update its env setting and recreate its containers; restart alone does
not load changed env. Direct Compose operations must include `.env`, `.env.local`
and `.release-images.env` and must not overlap deployment.

## Release workflow

A `v*` tag triggers deployment. Local `./deploy.sh -m "message"` only commits and
pushes notebooks, private backend and main checkout; it does not upload artifacts
or create a tag. Use `./deploy.sh patch|minor|major -m "message"` to also upload
models/metrics and create and push the release tag. Review these checkouts before
invoking either mode.

Actions fingerprints Docker inputs and lockfiles, reuses matching images and builds
missing identities. Clio/Prophet share one pinned private-backend commit. The input
map is in `scripts/deployment/release.py`; tests compare it against Dockerfiles.
Shared package changes rebuild their consumers. Documentation copied into images
also participates in the fingerprint. `rebuild=true` refreshes application base
images; infrastructure image upgrades are separate maintenance.

Actions uploads a release bundle and invokes its `deploy.sh /var/www`:

1. Lock deployment, validate Compose and download images.
2. Compare schema/config fingerprints with the last successful release; stop affected writers.
3. Before migrations, save `pg_dumpall` including roles to `/var/www/backups/` with mode 0600.
4. Install configs, Compose, image references and root `argus` with `scripts/prod/`;
   write `.argus-mode=prod`. Operator env, models and data remain separate.
5. Wait for infrastructure, apply changed domain migrations and reconcile services.
6. Reload nginx and record successful fingerprints.

Unchanged containers stay running unless shared mounted configuration requires a
restart. Actions serializes deployments; host flock excludes concurrent maintenance.
Phase timing in deploy logs identifies download, backup, migration and startup costs.

## Recovery

A failure stops deployment without advancing successful fingerprints. Affected
writers may remain stopped. Inspect the failing phase, correct it and rerun the
same release; migration checks and Compose reconciliation run again. The uploaded
bundle and pre-migration backup remain available.

There is no automatic database rollback. Restoring the full-instance dump affects
all domains and must account for writes made after the backup. Stop writers and
coordinate the database snapshot, application images and model artifacts before
restoring. Keep historical Alembic files in version control.

Use `/var/www/argus` from any directory; `/var/www/bin/argus` links to it.
See [commands](docs/commands.md) for logs/status and the service READMEs for
application-specific retries. Local orchestration tests use fake Docker; real
image builds and rollout are validated by the deployment workflow.
