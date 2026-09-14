# Prophet runtime and contracts

Prophet owns forecast execution, input snapshots, releases, exports and hourly
slots in its own PostgreSQL schema. `prophet` is the worker; `prophet-api` is the
internal HTTP reader. They share the same independently built image. Observations
are obtained through Clio HTTP, never by reading its tables.

## Runs, publication and reads

A run saves the exact input snapshot, source/dependency/model hashes and compressed
CSV artifacts. Model binaries are not archived; retain matching models for replay.
Run states are `running`, `succeeded`, `partial`, `failed`, `interrupted`.

A completed product contains all contract-declared sources from one run/issue time.
Run completion, complete product publication and scheduled-slot completion commit
in one transaction. Failed calculations preserve previous current releases. Optional
density skips permit other products to advance. Historical releases remain readable.

Authenticated contracts use `Authorization: Bearer <FORECASTS_SERVICE_TOKEN>`:

```text
GET /internal/v1/forecasts/{product}/latest
GET /internal/v1/forecasts/{product}/releases/{release_id}
GET /internal/v1/forecasts/{product}/status
```

`ForecastRelease` in `common.schemas.forecast_release` contains release/run UUIDs,
product, issue/publication times and exact CSV artifacts with hashes and metadata.
Products: solar-wind-speed, solar-wind-density, geomagnetic-activity, dst, hmf,
solar-radiation and atmospheric-density. The contract's presence does not enable
models absent from the current generation command (notably solar radiation).
`generate kp` calculates both Kp and Ap for the complete geomagnetic product.

Reads use repeatable-read snapshots. Missing current releases and storage failures
return 503, unknown products/historical UUIDs 404, invalid tokens 401. Payloads are
bounded to 32 MiB per artifact and 64 MiB per API response. API has no CSV/SQL
fallback. Model evaluation metrics remain static deployed files.

Health endpoints are unauthenticated and report infrastructure liveness/readiness,
not forecast fitness. [Diagnostic status](prophet-readiness.md) distinguishes the
published release from the latest attempt; unconfigured freshness limits are explicit.

## Scheduling and connection loss

The latest due hour runs at `:10 UTC`, with failure checks every 60 seconds. Missed
hours are not replayed. `forecast_slot` preserves attempt counts and completion;
`forecast_run.scheduled_slot` links attempts. Manual runs do not advance slots.
Partial runs complete their slot; export errors do not reopen it.

PostgreSQL session advisory lock `(736218, 2)` serializes supported writers across
workdirs/processes. They reuse that unpooled connection in short transactions;
there is no long transaction during model execution. Use direct or session-pooled
PostgreSQL, not transaction pooling. An old writer never reconnects silently after
session loss. The next owner marks abandoned runs/slots interrupted. CPU work can
finish before noticing disconnect; unique temporary files prevent collisions.

## Exports and operations

Live CSV is a retryable export of current database releases. Each file replacement
is atomic; the filesystem and database acknowledgement are not one transaction.
Before replacement the writer checks its lock session. Superseded pending releases
are not exported over newer ones. If a failure occurs around replacement, inspect
export status and restore current releases with `export --force` when needed.
There is no automatic retention for runs, releases or export archives.

Use the installed [host wrapper](../README_DEPLOY.md#operational-commands):

```bash
/var/www/bin/argus prophet runs --limit 10
/var/www/bin/argus prophet show-run <run-uuid>
/var/www/bin/argus prophet slots --limit 10
/var/www/bin/argus prophet status solar-wind-speed
/var/www/bin/argus prophet generate wind
/var/www/bin/argus prophet export
/var/www/bin/argus prophet export --force
```

Stop all manual writers before a deployment affecting their domain. Preserve
schema history and image/model artifacts during recovery; migrations are not a
substitute for a coordinated data restore. The scheduler's former filesystem
marker and transition-only CLI commands are no longer part of operation.
