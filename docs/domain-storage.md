# Domain storage

| Schema | Runtime | Owner/migrator | Data |
| --- | --- | --- | --- |
| api | argus_api | argus_api_migrator | Dashboard/authentication and API statistics |
| clio | argus_clio | argus_clio_migrator | Observations, aggregates, collector/source status and scheduling |
| intelligence | argus_intelligence | argus_intelligence_migrator | Processing attempts and per-release stub results |
| prophet | argus_prophet | argus_prophet_migrator | Runs, snapshots, artifacts, releases, exports and slots |

One PostgreSQL instance is shared, but SQL reads/writes remain within each domain.
Runtime credentials have DML privileges, no DDL, migration-marker updates, role
membership or foreign table/function access. Owner default privileges apply to
new domain objects. API reads observations and forecasts through owner HTTP
contracts, never foreign ORM models or SQL.

Each application owns its Alembic chain. **Keep historical migration files:**
they are required to create and upgrade databases, even when one-time migration
instructions and transition commands have been retired.

## Provisioning and maintenance

Administrative credentials are passed only to `db-bootstrap` and
`intelligence-provision`; migration passwords
only to maintenance services. Runtime containers do not receive either. Normal
release deployment does not run legacy bootstrap or rotate passwords. Intelligence
provisions its own domain before changed migrations; see its
[rollout requirements](../apps/intelligence/README.md).

For a new database, configure the eight domain passwords and admin settings,
start PostgreSQL and run the explicit maintenance sequence with pinned images:

```bash
argus compose run --rm --no-deps db-bootstrap
argus compose run --rm --no-deps clio-migrate
argus compose run --rm --no-deps api-migrate
argus compose run --rm --no-deps prophet-migrate
argus compose run --rm --no-deps intelligence-provision
argus compose run --rm --no-deps intelligence-migrate
```

The host must first have the release Compose/configuration and image references
installed as described in [deployment](../README_DEPLOY.md). The bootstrap remains
an operator tool for provisioning/grants/password maintenance. Stop relevant
writers and take a backup before administrative storage changes. It also retains
legacy-database compatibility for recovery; historical cutover instructions are
available in Git history, not repeated in current release workflows.

## Verification

The public CI `domain-storage` job runs isolated PostgreSQL tests covering fresh
installation, legacy upgrades, row/OID preservation, grants and rejected foreign
access, observation retention/aggregation, dashboard access, forecast publication,
slot atomicity and session-lock recovery.

Local integration tests require an explicitly disposable server configured through
`TEST_DATABASE_ADMIN_DSN`. They create/drop temporary databases and provision the
six reserved roles, so do not point them at production. They never fall back to
project database credentials. Without that variable those suites skip.

Boundary and deployment-planning tests can run without Docker or PostgreSQL:

```bash
.venv/bin/python -m pytest -q tests/test_architecture.py tests/test_domain_deployment.py tests/test_release_deployment.py
```
