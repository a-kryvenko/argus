# Incremental service extraction

## Agreed target

- Clio owns observation ingestion, storage, aggregates and read contracts.
- Prophet owns forecast execution, input snapshots, database releases and live CSV exports.
- Intelligence consumes explicit contracts; its initial implementation reports health only.
- API owns authentication and user-facing HTTP orchestration.
- One PostgreSQL instance, domain-owned schemas/migrations/runtime roles. No foreign table reads.

## Step 1: standalone Prophet runtime

Implemented first because observation storage is currently coupled to dashboard,
ingestion, normalized history and private calibration helpers. Moving forecasting
first establishes a read boundary without introducing shared database access.

`apps/prophet` contains the forecast commands, density input preparation, CLI,
worker and its independent environment. API contains the temporary observation
read endpoint; Prophet consumes it via HTTP with no database credentials. Existing
CSV readers and output formats are retained. The full run reads its normalized
and density source inputs in one request/transaction.

Production uses a persistent Compose worker instead of forecast crontab. The
single-host scheduler completion marker is temporary operational state, not the
planned run/release ledger. See [Prophet runtime](../apps/prophet/README.md).

## Step 2: Clio and observation ownership

Clio now owns collector orchestration, persistence, aggregation, diagnostics,
read endpoints and observation migrations in `apps/clio`. Separate collector
processes are preserved. Public API and dashboard observation reads use the Clio
HTTP interface. Prophet's observation URL points to Clio; its contract is unchanged.

The API/Clio schemas have separate runtime and migration roles. The operator-only
bootstrap adopts legacy tables and trigger functions without copying rows or
exposing compatibility views. See [database cutover](domain-storage.md).

Private calibrated observation preparation is isolated behind the Clio ingestion
adapter; private source stays private. Ingestion/aggregation cron commands are
replaced by owned Clio scheduler processes with database completion markers and
shared advisory locks. Existing collector/retention behavior is retained.

## Step 3a: Prophet run accounting (implemented)

Prophet owns its schema, runtime/migrator roles and migrations. Each execution saves
its input snapshot, model/source hashes, dependency versions and compressed CSV
artifacts before replacing live CSV files. Runs expose success, partial, failure
and interrupted states through the installed CLI. API still reads live CSV;
filesystem scheduling and locking remain unchanged. See [rollout](prophet-runs.md).

## Step 3b: Forecast publication (next)

Add independent per-product releases and their read contract. Publish completed
products transactionally in PostgreSQL; export the published release to live CSV
with a retriable export status. Move readers to Prophet contracts. Define explicit
freshness and data readiness policy. Replace the temporary filesystem scheduler
marker/lock with durable scheduling and execution coordination.

## Step 4: Intelligence stub

Add an independent app/image with liveness/readiness and contract version. Report
that calculations are not implemented; do not fabricate risk results. Keep future
impact inputs tied to exact forecast releases and orbital-data versions.
