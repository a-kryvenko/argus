# Intelligence worker (stub)

Intelligence polls the current `solar-wind-speed` release every 60 seconds through
Prophet's HTTP contract. It validates the full release and saves a stub result in
its own PostgreSQL schema. It does not calculate satellite risks or enforce model
readiness. No private backend is installed.

## Deployment: required action before creating the release tag

Add **two new, distinct random passwords** to the existing production env file:

| Variable | Used by |
| --- | --- |
| `INTELLIGENCE_DB_PASSWORD` | Runtime role `argus_intelligence` |
| `INTELLIGENCE_MIGRATION_PASSWORD` | Owner/migrator `argus_intelligence_migrator` |

Keep the existing `DB_NAME`, administrative credentials and
`FORECASTS_SERVICE_TOKEN`. No new GitHub secrets or scheduler settings are needed.
Compose requires both new values and fails validation before stopping services if
one is absent. The production env files are never rewritten by deployment.

When Intelligence migrations/provisioning change, deployment runs the isolated
`intelligence-provision` maintenance container before stopping writers. It creates
only the Intelligence roles/schema if missing and verifies the supplied passwords.
It does not adopt legacy data, modify other domains or rotate existing passwords.
An unexpected schema owner or elevated/member role is rejected. A wrong existing
password requires explicit rotation; it is not silently changed during deployment.

Deployment then stops Intelligence, backs up PostgreSQL, runs its Alembic chain
and starts the worker through Compose. Runtime receives only its own SQL password
and forecast token. The migrator receives only its migration password; admin
credentials are confined to the provisioning container. No cron entry is needed.

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

For local use, run `uv sync --project apps/intelligence --frozen`, configure the
same HTTP/runtime database variables (and migration password for migrations),
and use `./scripts/intelligence ...`. `intelligence migrate upgrade head` applies
only this domain's migrations after provisioning.

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

Password rotation is explicit maintenance: stop Intelligence, update the two env
values, run `argus compose run --rm --no-deps intelligence-provision
intelligence-provision --rotate-passwords`, and recreate Intelligence with the
updated environment. Coordinate this separately from deployments.
