# Intelligence worker (stub)

Intelligence polls the current `solar-wind-speed` release every 60 seconds through
Prophet's HTTP contract. It validates the full release and saves a stub result in
its own PostgreSQL schema. It does not calculate satellite risks or enforce model
readiness. No private backend is installed.

## Database and deployment

Intelligence has its own PostgreSQL database and one owner for both runtime and
Alembic. Set `INTELLIGENCE_DB_HOST`, `INTELLIGENCE_DB_PORT`,
`INTELLIGENCE_DB_NAME`, `INTELLIGENCE_DB_USER`, `INTELLIGENCE_DB_PASSWORD`.
Passwords are raw strings, with no URL encoding. Keep `FORECASTS_URL` and `FORECASTS_SERVICE_TOKEN` configured.

Create the database explicitly with the shared maintenance tool before the first
release. Normal deployment runs migrations only; it never creates roles/databases
or rotates passwords. See [database provisioning and transfer](../../docs/domain-storage.md).

## Commands

```bash
/var/www/bin/argus intelligence status
/var/www/bin/argus intelligence process
/var/www/bin/argus intelligence check
/var/www/bin/argus intelligence check dst --release-id <uuid>
```

`check` remains an HTTP-only diagnostic: it does not write the ledger. `process`
runs one accounted processing attempt, while `status [product]` returns the latest
attempt, latest saved result and number of identified pending releases. Products
other than solar-wind-speed can be processed manually; the production worker
command selects solar-wind-speed explicitly by default.

For local use, run `uv sync --project apps/intelligence --frozen`. The unified
wrapper selects the installed environment and loads the root env files:

```bash
./scripts/argus intelligence migrate upgrade head
./scripts/argus intelligence worker
```

## Processing and recovery

A dedicated PostgreSQL session lock serializes all workers and manual processing.
All writes use that same connection. Losing it cannot silently reconnect a writer.
Each poll records an attempt; repeated successful release IDs are marked skipped
without recomputing the stub result. Result insertion and attempt success commit
in one transaction, with a unique `(product, release_id)` key.

Unfinished attempts become interrupted when the next writer acquires the lock.
Identified failed/interrupted releases are retried by their exact release ID before
fetching the latest release. HTTP failures before identifying a release retry the
latest endpoint. Errors are sanitized; credentials and response bodies are not
stored in logs. SIGTERM requests a graceful stop after the current attempt.

The current HTTP contract exposes latest and ID-based reads, not a publication
feed. Therefore releases superseded between polls or during downtime are **not
backfilled**. An identified release that remains unavailable will remain pending
and block newer work for that product until the failure is resolved. The ledger
currently retains all attempts, including skipped polls; retention and retry
quarantine are future operational improvements.

`status` reports database history, not process liveness. `mode: stub` and
`risk_assessment: null` remain explicit in saved results. Processing success does
not mean the input is fresh or safe for operational risk decisions.

Password rotation is separate maintenance: stop the worker, change its owner's
password administratively, update `INTELLIGENCE_DB_PASSWORD`, and recreate the
container with `argus compose up -d intelligence`.
