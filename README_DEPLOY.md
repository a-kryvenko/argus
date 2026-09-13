# Deployment

The [deploy workflow](.github/workflows/deploy.yml) runs when a `v*` tag is pushed.
It builds and publishes the frontend/API/Clio/Prophet images, uploads deployment files,
installs the cron schedule, pulls images, applies database migrations and restarts
the Compose services.

## Prerequisites

- Production `.env` and `.env.local` configured in `/var/www`, including a strong
  random `OBSERVATIONS_SERVICE_TOKEN`, **`FORECASTS_SERVICE_TOKEN`**,
  and six separate domain passwords:
  `API_DB_PASSWORD`, `API_MIGRATION_PASSWORD`, `CLIO_DB_PASSWORD`,
  `CLIO_MIGRATION_PASSWORD`, `PROPHET_DB_PASSWORD`,
  `PROPHET_MIGRATION_PASSWORD`. See [database cutover](docs/domain-storage.md).
- GitHub secrets: `DOCKERHUB_LOGIN`, `DOCKERHUB_TOKEN`,
  `FORECAST_CORE_DEPLOY_KEY`, `PROD_HOST`, `PROD_USER`, `PROD_SSH_KEY`.
- The private forecast backend committed and pushed. CI checks out its `master`
  branch; the Clio and Prophet lockfiles must match that checkout.
- Model and metric artifacts uploaded to `/var/www/data/models` and
  `/var/www/data/metrics`.

## Prophet readiness diagnostics

**No new environment variables, migrations or dependencies in stage 3c.2a.**
The existing tokens and domain passwords remain required. Deployment no longer
runs the one-time `import-schedule` command, since production has completed that
cutover. Older installations must still follow [checkpoint adoption](docs/prophet-scheduling.md).

New runs save diagnostic evidence with their snapshots. Use `prophet status <product>`
to inspect current release age and the last calculation attempt. New blocking
thresholds are not enabled; the existing density age check remains. See
[readiness diagnostics](docs/prophet-readiness.md).

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
completed slots in PostgreSQL. Both domains use their own advisory locks; Prophet
commits slot completion together with forecast publication. CSV export retries
do not recalculate completed slots.

No application commands remain in [crontab](.deploy/cronjobs.txt); its only entry
starts the external nginx proxy after host reboot. No history deletion is scheduled.
The ten-minute forecast offset still does not guarantee observation readiness.

Manual work uses installed commands:

```bash
docker compose run --rm clio clio refresh
docker compose run --rm clio clio aggregate
docker compose run --rm prophet prophet generate
```

Finish independent manual jobs before deploying. The workflow stops service
writers and saves a PostgreSQL dump,
provisions/adopts domain storage, runs separate Clio/API/Prophet migrations, then restarts
the stack. Production has already completed the legacy ownership cutover;
[cutover and recovery](docs/domain-storage.md) documents adoption for older databases. Migration and
admin credentials exist only in maintenance-profile services, not runtime apps.

## Dashboard authentication

Before starting the updated API, apply `pnpm db:migrate` locally or run the
`api-migrate` maintenance service. Configure `DASHBOARD_ORIGINS`
with the exact HTTPS site origin and `DASHBOARD_COOKIE_SECURE=true`. Create the
first administrator using the interactive command described in
[Dashboard setup](docs/dashboard.md#setup).
