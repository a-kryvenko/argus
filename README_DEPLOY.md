# Deployment

The [deploy workflow](.github/workflows/deploy.yml) runs when a `v*` tag is pushed.
It builds and publishes the frontend/API/Clio/Prophet images, uploads deployment files,
installs the cron schedule, pulls images, applies database migrations and restarts
the Compose services.

## Prerequisites

- Production `.env` and `.env.local` configured in `/var/www`, including a strong
  random `OBSERVATIONS_SERVICE_TOKEN` and four separate domain passwords:
  `API_DB_PASSWORD`, `API_MIGRATION_PASSWORD`, `CLIO_DB_PASSWORD`,
  `CLIO_MIGRATION_PASSWORD`. See [database cutover](docs/domain-storage.md).
- GitHub secrets: `DOCKERHUB_LOGIN`, `DOCKERHUB_TOKEN`,
  `FORECAST_CORE_DEPLOY_KEY`, `PROD_HOST`, `PROD_USER`, `PROD_SSH_KEY`.
- The private forecast backend committed and pushed. CI checks out its `master`
  branch; the Clio and Prophet lockfiles must match that checkout.
- Model and metric artifacts uploaded to `/var/www/data/models` and
  `/var/www/data/metrics`.

## Release helper

`./deploy.sh -t vX.Y.Z -m "Release description"` uploads model/metric artifacts
through the SSH host alias `argus`, commits all pending changes in `notebooks`,
`packages/forecast-core` and the root repository, pushes them, then creates and
pushes the tag. Review all three checkouts before running it.

Without `-t`, the helper still uploads, commits and pushes, but does not create
the tag that triggers deployment.

## Scheduled work

[Compose](.deploy/docker-compose.yml) runs separate solar wind and Kp/Dst
collectors, `clio-refresh` (hourly at :00 UTC), `clio-aggregate` (every five minutes)
and `prophet` (hourly at :10 UTC). Each scheduler retries failures and persists
completed slots. Clio uses its database and shared advisory locks; Prophet still
uses its filesystem marker and output lock pending release accounting.

No application commands remain in [crontab](.deploy/cronjobs.txt); its only entry
starts the external nginx proxy after host reboot. No history deletion is scheduled.
The ten-minute forecast offset still does not guarantee observation readiness.

Manual work uses installed commands:

```bash
docker compose run --rm clio clio refresh
docker compose run --rm clio clio aggregate
docker compose run --rm prophet prophet generate
```

The workflow stops writers, drains old one-off jobs, saves a PostgreSQL dump,
provisions/adopts domain storage, runs separate Clio/API migrations, then restarts
the stack. This first ownership cutover requires downtime and legacy revision
`20260911_0010`; read [cutover and recovery](docs/domain-storage.md). Migration and
admin credentials exist only in maintenance-profile services, not runtime apps.

## Dashboard authentication

Before starting the updated API, apply `pnpm db:migrate` locally or run the
`api-migrate` maintenance service. Configure `DASHBOARD_ORIGINS`
with the exact HTTPS site origin and `DASHBOARD_COOKIE_SECURE=true`. Create the
first administrator using the interactive command described in
[Dashboard setup](docs/dashboard.md#setup).
