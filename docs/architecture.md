# Architecture

## Services and packages

| Component | Responsibility |
| --- | --- |
| `apps/api` | Public HTTP, dashboard authentication and API statistics; reads Clio, Prophet and Intelligence over HTTP |
| `apps/clio` | Observation collection, storage, aggregation, scheduling and internal reads |
| `apps/prophet` | Forecast generation, input snapshots, releases and scheduling |
| `apps/intelligence` | Consumes Prophet releases; serves on-demand LEO drag assessments and records worker stub results |
| `apps/web` | Next.js frontend |
| `packages/common` | Configuration and shared contracts |
| `apps/clio/src/clio/providers` | Public provider fetching and parsing, owned by Clio |
| `packages/forecast` | Public solar-wind inference, features and shared calculation primitives |
| `packages/forecast-core` | Private non-solar-wind models, training and calibration; uses the public forecast library |
| `packages/intelligence-core` | Proprietary impact calculations; separate private repository |

Data flows from providers through Clio → Prophet → Intelligence. API reads Clio, Prophet and Intelligence contracts. Reads never collect data or generate forecasts. Forecast
artifacts are stored in PostgreSQL; model evaluation metrics are static files.
Prophet publishes each product independently and retries only failed products
within each product’s current schedule slot. It does not export forecast files.

Clio deploys as HTTP plus one worker container. One scheduler dispatches temporary
executors with independent live lanes and bounded background concurrency. It
retains task locks until executors exit, so normalization cannot block collection.

Clio ingestion accesses `forecast_core.calibration` and `forecast_core.observations`
through lazy adapters;
its base dependencies exclude model runtimes. Prophet installs the `[models]`
extra of `forecast-core`; its HTTP read
path does not import the backend. API installs without private code; Intelligence depends on `intelligence-core`
and uses its `api` boundary for on-demand drag calculations. Neither imports
other applications' runtimes. `intelligence-api` reads the density release over
HTTP and needs no database credentials; the existing worker remains a separate
release-integration process.

## Database ownership

One PostgreSQL instance hosts four independent databases. Each service and its
Alembic migrations share one database owner. Provisioning revokes public database
access; cross-service reads use HTTP. Resources and server availability remain shared.

| Service | Suggested database / owner | Schema | Migrations |
| --- | --- | --- | --- |
| API | `argus_api` | `api` | `apps/api/alembic` |
| Clio | `clio` | `clio` | `apps/clio/src/clio/migrations` |
| Prophet | `argus_prophet` | `prophet` | `apps/prophet/src/argus_prophet/migrations` |
| Intelligence | `argus_intelligence` | `intelligence` | `apps/intelligence/src/argus_intelligence/migrations` |

Each application receives only its own prefixed database credentials. Session
advisory locks serialize supported writers; Prophet and Intelligence reuse the
lock connection for writes and require direct or session-pooled PostgreSQL.

## Dependencies and configuration

Python dependency boundaries (distinct from HTTP service calls):

| Consumer | Internal package dependencies |
| --- | --- |
| `forecast` | `common` only |
| API | `common` only; calls Prophet and Intelligence over HTTP without installing them |
| Intelligence | `common`, private `intelligence-core`; calls Prophet over HTTP |
| `forecast-core` | `forecast`, `common` |
| Prophet | `forecast`, `forecast-core`, `common` |

Only Clio imports its `clio.providers` modules. Other services
receive observations over HTTP. Clio owns archive loading and caching; the private
backend exposes calibration computations over supplied data. Libraries must not depend on
application runtimes. Architecture tests check imports, declared dependencies
(including extras), and the API lockfile; private manifest checks run when its
checkout is available.

All packages live in `packages/`. Proprietary `forecast-core` and
`intelligence-core` are excluded from the public Git repository and maintained
in separate repositories. Both backend repositories remain private; numerical impact implementation and
its documentation belong in `intelligence-core`.
The public Python workspace resolves without private repositories. Solar-wind
speed and density inference run entirely in `forecast`, including feature building
and quantile blending. The private backend depends on that public library; the
public library never imports private code. Prophet's product catalog resolves
public and private adapters lazily, without a second model enumeration.
See the [public package example and tests](../packages/forecast/README.md).
`packages/forecast-core` is an ignored, separate Git checkout required by Clio
and Prophet builds. Keep its version consistent with both application lockfiles;
push private changes before deploying dependent public code. CI pins one private
commit for both images. Do not publish private implementation or training code.

Configuration uses `ARGUS_WORKDIR`, or searches the current directory and parents
for `configs/project.yaml`. The command adapters set the workdir and load root env
files. See [local setup](../README_DEPLOY.md#local-development),
[commands](commands.md) and [deployment](../README_DEPLOY.md).

## Validation

`./scripts/test-python -q` runs the application, public package and private backend
suites. Private tests require the private checkout and installed dependencies.
For domain integration, run `./scripts/test-domain-storage` with uv, Docker and
both private checkouts available. This is the exact entry point used in CI and
before deployment. It creates and removes a disposable PostgreSQL 17 container,
checks service imports and migrations first, then runs all domain integration
checks. An explicit `TEST_DATABASE_ADMIN_DSN` can instead select an existing
**disposable** PostgreSQL server with database/role administration rights.
The runner uses temporary configuration and disables Sentry; it does not load
local `.env` credentials.

`tests/runtime/pyproject.toml` depends on the four service packages and pytest.
Service libraries are declared only in their own manifests; there is no second
list in the workflow. `tests/runtime/uv.lock` fixes the combined integration
environment, independently of each service's deployment lock. After changing
service dependencies, update that service's lock and run
`uv lock --project tests/runtime`. The runner uses `uv sync --locked`, so stale
metadata fails during setup rather than producing dozens of test failures.
No service virtualenv or inherited `PYTHONPATH` is used.

Public boundary checks run without private credentials. Domain integration uses
private checkouts and therefore skips fork and Dependabot pull requests; it runs
on same-repository pull requests targeting `master`. Deployment always runs it
with the exact private commits selected for the release and waits for success.

Boundary tests check import direction, private adapters, absence of foreign-domain
SQL and isolated credentials. Integration tests cover migrations, ownership,
publication and locking. See each service README for its runtime guarantees.

Prophet keeps exclusive writer sessions in its coordinator. Spawned per-product
calculations receive saved snapshots and explicit model configuration, then return
serialized results; they never inherit a database connection. All backends use
`forecast_snapshot`, including private IMF and atmospheric density; density snapshot
preparation belongs entirely to `forecast-core`. GONG inputs
are owned and archived by Clio, and included in saved Prophet snapshots. Private
inference must not read project configuration or local observation files.

Prophet dispatches products on independent configurable schedules, with bounded
calculation concurrency and parent-owned generation writes. Verification uses a
separate process and writer lock; cleanup acquires both locks. Existing model
issue times and grids remain hourly even when execution schedules are more frequent.
