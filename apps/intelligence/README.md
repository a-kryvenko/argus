# Intelligence worker (stub)

Intelligence polls the current `solar-wind-speed` release every 60 seconds through
Prophet's HTTP contract. It validates the full release and saves a stub result in
its own PostgreSQL schema. It does not calculate satellite risks or enforce model
readiness. No private backend is installed.

## Configuration and operation

Configure `INTELLIGENCE_DB_*`, `FORECASTS_URL` and `FORECASTS_SERVICE_TOKEN`.
See [local setup](../../README_DEPLOY.md#local-development),
[commands](../../docs/commands.md) and [deployment](../../README_DEPLOY.md).
`./argus intelligence refresh [product]` runs one processing cycle;
`./argus intelligence status [product]` reports persisted processing history.
Without a product, these commands cover all supported products; the production
worker polls solar-wind-speed. Production Compose starts that worker automatically.

For local worker development use `./scripts/dev/run intelligence worker`.
The internal adapter's `intelligence check [product] [--release-id UUID]` validates
HTTP release retrieval without writing a ledger entry. Its `process` command is
the underlying one-cycle operation exposed as `refresh`.

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


Related document: [Atmospheric density API](jb2008-api.md).
