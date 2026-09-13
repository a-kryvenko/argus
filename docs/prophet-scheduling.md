# Prophet database scheduling: stage 3c.1

The initial marker adoption described below is complete on production. Current
deployments no longer repeat it; see [readiness diagnostics](prophet-readiness.md).

## Production rollout

**No new environment variables, secrets, roles or dependencies are required.**
The existing Prophet runtime/migration passwords and forecast/observation tokens
remain in use. `PROPHET_STATE_DIR` no longer controls active scheduling or locking;
it is consulted only by the explicit legacy-marker import command.

Production has already completed the stage 3b publication cutover. The workflow
therefore replaces its one-time `publish-existing` step with `import-schedule`.
After stopping current workers and finishing independent manual jobs, it backs up
PostgreSQL, runs domain migrations, then executes:

```bash
docker compose run --rm --no-deps prophet prophet import-schedule
docker compose up -d --remove-orphans
```

The existing `prophet-migrate` step applies `20260913_prophet_scheduler` first.
Do not run the old filesystem-lock worker alongside the new database-lock worker:
the two lock mechanisms cannot exclude each other. The same applies to old manual
`generate` and `export` processes. Finish them before deploying.

`import-schedule` reads `data/prophet/last-completed-slot` under `ARGUS_WORKDIR`.
If the previous worker used a different state directory, supply its actual marker:

```bash
docker compose run --rm --no-deps prophet prophet import-schedule --marker /var/www/data/custom-state/last-completed-slot
```

The command does not alter or delete the old file. It imports the completed UTC
hour as a slot with status `imported` and zero attempts; it does not invent run
history or releases. Repeating it does not overwrite an existing slot. Missing
files produce a warning and no checkpoint, appropriate for a fresh installation;
in that case the latest due slot may run once. Invalid, non-hourly or future
markers fail explicitly before startup. Import while old writers are stopped.

Fresh installations need only migrations and the normal worker. For an older
stage 3a installation that has not yet populated release pointers, additionally
follow the [publication cutover](prophet-publication.md) before switching API reads.
This extra step is not needed on the already upgraded production server.

## Execution semantics

One PostgreSQL session advisory lock `(736218, 2)` serializes supported Prophet
writers across processes: scheduled runs, manual generation, CSV export and
administrative publication/import commands. It is database-wide, independent of
workdir or state-directory paths. Observation collectors and Clio scheduling keep
their existing separate processes and locks.

The lock connection is unpooled and stays open throughout a command. Database
operations use short transactions on that same connection, leaving it in
autocommit between operations. There is no long transaction spanning model
computation. Use a direct PostgreSQL connection (as in Compose) or session pooling;
transaction pooling is incompatible with this session lock.

`prophet.forecast_slot` records each attempted UTC hour, its status, attempt count,
last attempt start/finish and error. `forecast_run.scheduled_slot` links every new
scheduled attempt to its hour. Old run records keep a null slot, because their
exact scheduler slot cannot reliably be inferred from model or run timestamps.

Scheduling still runs the latest due hour at `:10 UTC`, polls every 60 seconds,
retries failures and skips completed (`succeeded`, `partial`, `imported`) slots.
An outage spanning several hours results in the latest due slot only; missed
hours are not replayed. A failed older slot remains in history if a later hour
becomes due. Clock rollback does not rerun hours below the completed high-water
mark. Manual generation does not advance the scheduled checkpoint.

Run completion, publication pointers and slot completion commit in one database
transaction. A crash after commit cannot lose the completion marker and trigger
a duplicate attempt for that slot. A failure before commit leaves that attempt
incomplete; the next lock owner marks abandoned running attempts/slots
`interrupted`. Retrying increments the slot attempt count and creates a new run
UUID, preserving the previous attempt and its error.

An optional density skip completes the slot as `partial`, as before. CSV export
runs separately and retries even when the scheduled slot is already complete.
An export error cannot reopen a successful slot or force model recalculation.
SIGTERM still lets the active operation finish within the Compose grace period.

## Lost connections and CSV limits

A disconnected writer never reconnects inside its existing command. Its next
database operation fails on the original session, so it cannot publish after a
new writer obtains the lock. PostgreSQL releases the lock when that session ends;
an interrupted run is recovered when another writer acquires it. A model may
finish CPU work before noticing session loss. Unique temporary files prevent
that abandoned calculation from overwriting another attempt's temporary output.

Before replacing a live CSV, export checks the lock session again. CSV replacement
and database acknowledgement remain separate operations, not a distributed
transaction. If a connection fails around file replacement, inspect export status
and use `prophet export --force` to restore current database releases when needed.
HTTP readers continue using committed database releases independently of CSV.
The deployment still uses its shared live-data volume; this stage does not add
per-host exports, a task queue or distributed filesystem guarantees.

## Verification and operations

```bash
docker compose run --rm prophet prophet slots --limit 10
docker compose run --rm prophet prophet runs --limit 10
docker compose run --rm prophet prophet show-run <run-uuid>
docker compose run --rm prophet prophet export
```

After the initial import, `slots` should show the old completed hour as `imported`.
The next due hour should have a linked run and finish as `succeeded` or `partial`.
Restarting the worker within the same slot must not add another attempt. The
`started_at` and `finished_at` fields on slots describe the latest attempt;
`runs`/`show-run` preserve individual attempts and their `scheduled_slot`.

PostgreSQL integration tests cover independent competing sessions, terminated
lock connections, restart before/after publication commit, failure retries,
manual/scheduled separation, export failures and marker adoption. They run in the
public `domain-storage` CI job against an isolated server. Locally they require
`TEST_DATABASE_ADMIN_DSN`; they never use production/project database credentials.

## Rollback and next substep

The migration adds a table and nullable run linkage; it does not rewrite existing
results or release IDs. Keep these additions when rolling application images
back. Stop all new writers before starting an older worker. The old filesystem
checkpoint is no longer updated, so rollback to it can repeat the latest due
hour once. Do not run old and new schedulers simultaneously. Downgrading deletes
slot history/linkage and is not required for an application rollback.

Readiness, maximum observation/forecast ages and product-specific gating remain
unchanged. They are the next substep, 3c.2; no new freshness thresholds are hidden
in this scheduler change.
