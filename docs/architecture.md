# Architecture

## Services and packages

| Component | Responsibility |
| --- | --- |
| `apps/api` | Public HTTP, dashboard authentication and API statistics; reads Clio and Prophet over HTTP |
| `apps/clio` | Observation collection, storage, aggregation, scheduling and internal reads |
| `apps/prophet` | Forecast generation, input snapshots, releases and scheduling |
| `apps/intelligence` | Consumes Prophet releases; records attempts and deduplicated stub results |
| `apps/web` | Next.js frontend |
| `packages/common` | Configuration and shared contracts |
| `packages/clio` | Public provider fetching and parsing |
| `packages/forecast` | Public solar-wind inference, features and shared calculation primitives |
| `packages/forecast-core` | Private non-solar-wind models, training and calibration; uses the public forecast library |
| `packages/intelligence-core` | Proprietary impact calculations; separate repository not yet created |

Data flows from providers through Clio → Prophet → Intelligence. API reads Clio
and Prophet contracts. Reads never collect data or generate forecasts. Forecast
artifacts are stored in PostgreSQL; model evaluation metrics are static files.
Prophet publishes each product independently and retries only failed products
within the current hourly slot. It does not export forecast files.

Clio deploys as HTTP plus one worker container. The worker supervises the existing
collector and scheduler processes; it adds no queue or storage model. Each loop
keeps its own database locks and timing, so normalization cannot block collection.

Clio ingestion accesses private calibration through a lazy adapter; its HTTP read
path does not import the backend. API and Intelligence install without private
code and cannot import other applications' runtimes. Private impact calculations
are not integrated; `packages/intelligence-core` is not an application dependency.

## Database ownership

One PostgreSQL instance hosts four independent databases. Each service and its
Alembic migrations share one database owner. Provisioning revokes public database
access; cross-service reads use HTTP. Resources and server availability remain shared.

| Service | Suggested database / owner | Schema | Migrations |
| --- | --- | --- | --- |
| API | `argus_api` | `api` | `apps/api/alembic` |
| Clio | `argus_clio` | `clio` | `apps/clio/src/argus_clio/migrations` |
| Prophet | `argus_prophet` | `prophet` | `apps/prophet/src/argus_prophet/migrations` |
| Intelligence | `argus_intelligence` | `intelligence` | `apps/intelligence/src/argus_intelligence/migrations` |

Each application receives only its own prefixed database credentials. Session
advisory locks serialize supported writers; Prophet and Intelligence reuse the
lock connection for writes and require direct or session-pooled PostgreSQL.

## Dependencies and configuration

All packages live in `packages/`. Proprietary `forecast-core` and
`intelligence-core` are excluded from the public Git repository and maintained
in separate repositories. The repository for `intelligence-core` has not been
created yet; its local source stays private.
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
Integration tests require `TEST_DATABASE_ADMIN_DSN` pointing to a disposable
PostgreSQL server; they create and delete random databases and roles. Without it,
database tests skip rather than use project credentials. Public CI provides PostgreSQL 17.

Boundary tests check import direction, private adapters, absence of foreign-domain
SQL and isolated credentials. Integration tests cover migrations, ownership,
publication and locking. See each service README for its runtime guarantees.
