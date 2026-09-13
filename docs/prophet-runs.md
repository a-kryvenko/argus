# Prophet run accounting: stage 3a (historical)

This page describes the original ledger rollout. For the current deployment and
behavior, use [stage 3b publication](prophet-publication.md).

## Required production preparation

**Before releasing, add two new required variables to `/var/www/.env.local`:**

| Variable | Used by |
| --- | --- |
| `PROPHET_DB_PASSWORD` | Runtime role `argus_prophet` |
| `PROPHET_MIGRATION_PASSWORD` | Schema owner `argus_prophet_migrator`, maintenance only |

Use separate strong secrets. There are no defaults or administrative credential
fallbacks. Existing `DB_NAME` and database host settings are passed by Compose.
The runtime receives no migration password and cannot read Clio/API tables.

Finish independent manual jobs before starting a deployment.
The tag-triggered workflow checks the incoming Compose configuration before
replacing installed configuration or stopping workers. It then pulls images,
stops service writers, backs up PostgreSQL, runs `db-bootstrap`, `clio-migrate`,
`api-migrate`, `prophet-migrate`, and starts the stack. Repeatable bootstrap adds
the third domain to an already adopted database without moving existing tables.

For a manual rollout, use the new images and Compose file, configure the Compose
environment files, stop/drain writers and complete the backup first. Then run:

```bash
docker compose run --rm db-bootstrap
docker compose run --rm clio-migrate
docker compose run --rm api-migrate
docker compose run --rm prophet-migrate
docker compose up -d
```

Commands assume the production Compose project directory and its environment
are selected. Do not start the new Prophet worker before provisioning/migration.
The workflow performs this sequence automatically.

## Verification

```bash
docker compose run --rm prophet prophet runs --limit 10
docker compose run --rm prophet prophet show-run <run-uuid>
```

An empty list immediately after rollout is expected: historical CSV files are
not imported and completed hourly scheduler markers are preserved. Wait for the
next scheduled slot, or deliberately run `docker compose run --rm prophet prophet
generate` to calculate now. `show-run --inputs` also displays the saved observations.
A successful run has stored artifacts with checksums and CSV acknowledgement times.
Optional unavailable density produces a `partial` run with a recorded skip reason.

## Stored evidence and failure behavior

`prophet.forecast_run` holds the execution UUID, product, trigger, scope, timestamps,
status, observation snapshot and hash, dependency versions and source-code hashes.
`prophet.forecast_artifact` stores each exact CSV as gzip-compressed bytes, its
SHA256, row count, column names and model metadata. Loaded model bytes are hashed;
model binaries are not archived. Retain matching model artifacts/source versions
for later replay. This stage does not implement automatic or bit-for-bit replay.

The input snapshot is written once through the recorder before calculation. Each
result is persisted before its live CSV is replaced. If that database write fails,
that file retains its previous contents; earlier products may already have changed.
`csv_written_at` is acknowledged after replacement. A missing acknowledgement can
also mean a crash between replacement and acknowledgement; database and filesystem
updates are not one transaction. CSV serving remains unchanged.

Errors are recorded as `failed` when the database is available. After an abrupt
stop, the next generation under the shared workspace lock marks previous running
attempts in that scope as `interrupted`. Recovery is not immediate. This assumes
the existing single-host deployment and shared lock volume, not distributed workers.
A successful or partial calculation keeps the existing scheduler completion rules.

There is no retention job for runs or artifacts yet; storage grows with executions.
Compressed CSV storage is an archive, not a per-timestamp SQL forecast interface.
The migration creates new Prophet tables; it does not rewrite existing forecasts.
Downgrading it deletes this history. Back up before rollback; older workers can
still use CSV but will not record runs.

## Next stage

Introduce per-product published releases, an explicit Prophet read contract and
retriable CSV export, then durable database scheduling/coordination. Intelligence
will consume that contract. The current ledger is not yet a published-release API.
