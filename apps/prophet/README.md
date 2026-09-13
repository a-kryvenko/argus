# Prophet runtime

Prophet owns forecast execution. It has an independent Python environment and
Docker image. It reads observations only through the versioned owner HTTP
contract and writes the existing live forecast CSV products. It records runs,
input snapshots and compressed results in its own PostgreSQL schema.

Clio owns observations and serves the read contract. API consumes Clio too and
has no observation SQL models. Prophet still publishes CSV files; database-backed
publication pointers and forecast read contracts belong to the next extraction stage.

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
revisions are still possible. Persisted input snapshots and replayable release
identifiers are distinct from the input snapshot saved in each Prophet run.

## Scheduling and limitations

The Compose `prophet` worker replaces the forecast cron entry. It runs the latest
due hourly slot (at :10 UTC), retries failures every 60 seconds, and remembers
successful slots under `data/prophet`. Restart skips completed slots and catches
up only the latest due slot, without replaying every missed hour. Actual forecast
issue times retain existing model behavior; scheduler slots are not release IDs.
The worker reads the latest committed observations; data-readiness gating and
staleness policies are not introduced in this extraction.

A shared-volume `flock` serializes worker and CLI generation, preventing competing
writes to live CSV files. An abrupt stop after output but before the marker can
cause regeneration; execution is not exactly once. This mechanism is for the
current single-host deployment. Execution records now live in PostgreSQL; leases and
per-product publication remain for the later publication stage.

A missing optional density input retains existing behavior: other products are
published and density is skipped. A failure halfway through the product list can
leave products from different issue times, as before. CSV publication is atomic
per file, not across a complete release. Manual generation does not advance the
scheduler marker. SIGTERM allows the current calculation to finish within the
Compose stop grace period.

## Run accounting and deployment

**New required production variables: `PROPHET_DB_PASSWORD` and
`PROPHET_MIGRATION_PASSWORD`. Set both before releasing this version.**
Runtime uses only the first password; the maintenance migrator uses only the second.
Local generation also requires an initialized Prophet database and runtime credentials.

The deploy workflow validates the new Compose configuration before replacing the
installed configuration or stopping workers. After backup and writer shutdown it
runs bootstrap, Clio/API/Prophet migrations, then restarts services.

```bash
./scripts/prophet runs --limit 10
./scripts/prophet show-run <run-uuid>
./scripts/prophet show-run <run-uuid> --inputs
```

See [run accounting and rollout](../../docs/prophet-runs.md) for the exact changes,
production verification, failure behavior and remaining publication work.
