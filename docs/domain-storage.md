# Domain database ownership and cutover

## Ownership

| Schema | Runtime role | Owner/migration role | Data |
| --- | --- | --- | --- |
| `clio` | `argus_clio` | `argus_clio_migrator` | Measurements, normalized observations, solar wind/Kp/Dst, source status, aggregates, pending/retired hours, scheduler slots |
| `prophet` | `argus_prophet` | `argus_prophet_migrator` | Forecast runs, input snapshots, compressed forecast artifacts, provenance, published releases and export status |
| `api` | `argus_api` | `argus_api_migrator` | Dashboard users/groups/memberships/sessions, login attempts, API statistics |

Each schema has its own Alembic version table and migration history. Runtime
roles get data read/write privileges and sequence usage but no schema creation,
DDL, migration-marker changes, role membership or foreign schema/table/function
access. Future objects inherit domain-specific grants from their migration role.
PostgreSQL privileges enforce the boundary even for SQL issued outside ORM code.
Prophet reads observations through Clio HTTP; its SQL credentials access only its own schema.

Database models use explicit domain schemas. Raw Clio SQL runs with the Clio
search path, and trigger functions explicitly set their own search path so their
behavior does not depend on the caller. There are no compatibility views or
shared ORM imports between applications.

## Existing databases

The operator-only `scripts/bootstrap-domain-db.py` adopts databases at legacy
revision **20260911_0010**. Older databases must first be upgraded using the
previous application release. Do not point the new domain migrators at a legacy
public schema; they deliberately refuse it.

Before adoption, stop all service and one-off writers, including legacy cron
aggregation, and make a PostgreSQL backup. This cutover requires downtime.
For a legacy server, remove old application cron entries and finish all one-off
jobs before deploying. The deployment workflow stops current service writers
and writes a custom-format dump under
`/var/www/backups/` before provisioning and migration. Stop any independent manual
processes outside that deployment as well.

The bootstrap checks all expected tables and the exact legacy revision. In one
transaction it provisions roles, moves tables (including their indexes,
constraints and owned sequences) to their schemas, transfers ownership, moves
and binds the trigger functions, creates domain version markers and applies
privileges. Rows are not copied or deleted. The old version marker is preserved
as `api.legacy_alembic_version`. Clio begins at the old observation head
`20260908_0009`; API begins at `20260911_0010`. Clio then applies its new scheduler
migration. A mismatch or DDL error rolls back the bootstrap transaction.

The API migration chain now starts with the dashboard migration; the observation
chain lives exclusively under `apps/clio/src/argus_clio/migrations`. This history
split is supported only through the explicit bootstrap, not by stamping or
running a normal upgrade against the legacy version table.

Bootstrap is repeatable after successful adoption: it refreshes grants/passwords
without moving data again. Reserved roles must not have pre-existing memberships
or own the database. Separate administrative credentials are used only by the
operator/bootstrap container; they are never passed to API, Clio or Prophet runtime.

## Production commands

Configure all six domain passwords (including the new `PROPHET_DB_PASSWORD` and
`PROPHET_MIGRATION_PASSWORD`) plus `OBSERVATIONS_SERVICE_TOKEN` and `FORECASTS_SERVICE_TOKEN` before the
release; Compose validates their presence. Maintenance services have separate
credentials and are excluded from normal `up` by their profile:

```bash
docker compose run --rm db-bootstrap
docker compose run --rm clio-migrate
docker compose run --rm api-migrate
docker compose run --rm prophet-migrate
docker compose run --rm --no-deps prophet prophet publish-existing
docker compose up -d
```

For a dry-run bootstrap, override its command:

```bash
docker compose run --rm db-bootstrap .venv/bin/python /var/www/scripts/bootstrap-domain-db.py
```

These commands assume writers are stopped and the backup is complete. The deploy
workflow performs that sequence. A new empty database follows the same sequence;
the domain migrators create their tables from scratch.

After a successful bootstrap, an old API/collector image is incompatible with the
new schema layout. If a later migration/startup fails, keep writers stopped and
fix forward, or restore the pre-cutover dump together with the previous release.
The bootstrap transaction and subsequent domain migrations are separate
transactions. Do not try to undo schema ownership by downgrading old migrations.

## Verification

The public CI `domain-storage` job provisions an isolated PostgreSQL 17 server.
It tests fresh installation and legacy adoption, OID/row preservation, repeatable
bootstrap, sequence allocation, trigger behavior under another search path,
retention guards, rejected foreign reads/DDL/role switching, scheduler restart
and locking, and existing aggregation/recovery/dashboard integration scenarios.

To run against your own disposable PostgreSQL server, explicitly configure
`TEST_DATABASE_ADMIN_DSN` with administrative permissions and run:

```bash
./scripts/test-python -q
```

The root test environment needs psycopg and Alembic for this integration suite;
CI installs them explicitly. Tests create temporary databases and provision the
six reserved roles, so this must be a separate test server. Without this variable
the database tests skip, and the ordinary contract/unit tests run normally.
Existing explicit integration scripts now also require this test DSN; they never
fall back to project database credentials.
