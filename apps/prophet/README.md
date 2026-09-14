# Prophet runtime

Prophet owns forecast execution. It has an independent Python environment and
Docker image. It reads observations only through the versioned owner HTTP
contract and writes the existing live forecast CSV products. It records runs,
input snapshots and compressed results in its own PostgreSQL schema.

Clio owns observations and serves the read contract. API consumes Clio too and
has no observation SQL models. API now reads published forecasts through the
Prophet HTTP contract; live CSV files are retryable exports of database releases.

## Local use

With the private `packages/forecast-core` checkout present:

```bash
uv sync --project apps/prophet --frozen
```

Configure `OBSERVATIONS_URL` (locally `http://127.0.0.1:8001`) and the same
`OBSERVATIONS_SERVICE_TOKEN` in the Clio and Prophet environments.
The token must be a strong random secret. The service refuses requests if it
is absent; there is no database or provider-download fallback.

```bash
./scripts/prophet generate
./scripts/prophet generate density
./scripts/prophet worker
./scripts/prophet serve --port 8002
./scripts/prophet export
```

Products for `generate`: `all`, `wind`, `kp`, `hmf`, `density`. The installed CLI
is the supported entry point; do not invoke implementation modules directly.
Set `ARGUS_WORKDIR` when running outside the checkout. Configuration/model/data
paths remain compatible with the existing deployment.

## Read contract

`GET /internal/v1/observations/forecast-inputs?as_of=<timezone-aware timestamp>`
requires `Authorization: Bearer <OBSERVATIONS_SERVICE_TOKEN>` and returns the
`common.schemas.forecast_inputs.ForecastInputs` version 1 contract:

- Normalized observations in the inclusive interval `[as_of - 30 days, as_of]`.
- Raw density source metrics in `[as_of - 88 days, as_of]`.
- `as_of` and `read_at`, both timezone-aware.

Both datasets are read in one PostgreSQL repeatable-read, read-only transaction.
A single full generation shares this response with density calculation. The
response is bounded to 250,000 raw measurements; an oversized response fails
explicitly rather than silently truncating. Database statements time out after
60 seconds; the client has a 180-second read timeout and a 10-second connect timeout.

`read_at` is the request's read-start time, not a durable snapshot ID. Historical
revisions are still possible. Prophet saves the exact response in its execution
record; published products reference that run.

## Scheduling and limitations

The Compose `prophet` worker replaces the forecast cron entry. It runs the latest
due hourly slot (at :10 UTC), retries failures every 60 seconds, and remembers
successful slots in PostgreSQL. Restart skips completed slots and catches
up only the latest due slot, without replaying every missed hour. Actual forecast
issue times retain existing model behavior; scheduler slots are not release IDs.
The worker reads the latest committed observations; data-readiness gating and
staleness policies are not enabled.

A PostgreSQL session advisory lock serializes worker, manual generation and CSV
export. Each hour and its attempts are recorded in `prophet.forecast_slot`; slot
completion commits with release publication. A restart after commit skips that
hour. Abandoned attempts become `interrupted` when the next writer gets the lock.
All writes use the lock's connection; session loss cannot silently reconnect an
old writer without its lock. See [database scheduling](../../docs/prophet.md).

A missing optional density input retains existing behavior: other products are
published and density is skipped. A failed calculation does not advance publication
pointers. CSV export is atomic
per file, not across a complete release. Manual generation does not advance the
scheduled slot. SIGTERM allows the current calculation to finish within the
Compose stop grace period.

## Run accounting and deployment

Production uses the selective [deployment workflow](../../README_DEPLOY.md).
Existing service tokens and domain database credentials remain required.
Use `/var/www/bin/argus prophet ...` on the server; the commands below are for
the local checkout.

```bash
./scripts/prophet status solar-wind-speed
./scripts/prophet slots --limit 10
./scripts/prophet runs --limit 10
./scripts/prophet show-run <run-uuid>
./scripts/prophet show-run <run-uuid> --inputs
```

See [Prophet operations](../../docs/prophet.md) for deployment and
recovery, and [publication contracts](../../docs/prophet.md) for reads and exports.

## Readiness diagnostics

New runs save input-age and data-quality evidence with their snapshots.
`prophet status <product>` and the authenticated `/internal/v1/forecasts/{product}/status`
endpoint distinguish the current release from the latest calculation attempt.
No new thresholds block generation or serving. See [diagnostic scope and next
policy decisions](../../docs/prophet-readiness.md).
