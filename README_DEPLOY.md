# Deployment

The [deploy workflow](.github/workflows/deploy.yml) runs when a `v*` tag is pushed.
It builds and publishes the frontend/API images, uploads deployment files,
installs the cron schedule, pulls images, applies database migrations and restarts
the Compose services.

## Prerequisites

- Production `.env` and `.env.local` configured in `/var/www`.
- GitHub secrets: `DOCKERHUB_LOGIN`, `DOCKERHUB_TOKEN`,
  `FORECAST_CORE_DEPLOY_KEY`, `PROD_HOST`, `PROD_USER`, `PROD_SSH_KEY`.
- The private forecast backend committed and pushed. CI checks out its `master`
  branch; the API lockfile must match that checkout.
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

[Compose](.deploy/docker-compose.yml) runs solar wind and Kp/Dst collectors
continuously. [Cron](.deploy/cronjobs.txt) runs:

| Schedule | Command |
| --- | --- |
| Every five minutes | `aggregate_solar_wind` |
| Hourly at :00 | `refresh_observations` |
| Hourly at :10 | `generate_forecast` |

The hourly jobs are independent; the ten-minute offset does not guarantee that
ingestion has finished. Forecasts read the latest committed observations.

The workflow replaces the server user's crontab with this file. Collector
healthchecks report loop progress; see [status diagnostics](docs/observation-status.md).
No scheduled history deletion is configured.

When switching aggregation worker versions, let older jobs finish before running
the new worker. Apply migrations before manually starting application commands.
