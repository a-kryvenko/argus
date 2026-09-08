# Package boundaries

| Package | Responsibility | Dependencies within the project |
| --- | --- | --- |
| common | Shared contracts, configuration, serialization, schema validation | None |
| clio | Fetch and parse source observations | common |
| forecast | Forecast interfaces, model calls and output files | common; optional local imports from `forecast_core.api` |
| forecast-core | Private forecasting, feature preparation, aggregation and calibration | common, clio |
| intelligence-core | Private satellite and power-grid impact calculations | Shared contracts as needed |
| api | HTTP, authentication, storage and scheduled commands | common, clio, forecast; private packages through their `api` modules |

Database models and migrations belong to API storage. Model input transformations
belong to forecast-core. Provider parsing belongs to clio. Atmospheric density is
a forecast product; satellite-specific impact belongs to intelligence-core.

The public `forecast` registry imports without private packages. Forecasts that
need a private backend use ordinary local imports from `forecast_core.api` and
raise `ModuleNotFoundError` if it is missing. API imports private services directly
from that module.
Public code must not import private implementation modules.
Backend source, model details, training notebooks and implementation-specific
tests must remain in private checkouts.

## Checkouts and installation

The public Python workspace contains common, clio and forecast. It must resolve
without private repositories. The root environment also supports research tools.

`packages/forecast-core` is a separate ignored Git checkout. The API environment
uses that checkout explicitly via a path dependency. Keep its source and version
consistent with `apps/api/uv.lock`. After updating it, run:

```bash
uv lock --project apps/api
uv sync --project apps/api --frozen
```

The local intelligence package is preserved under `private/intelligence-core`,
which is ignored by the public repository. It is not yet used by API. Transfer
this directory to a dedicated private repository before developing or deploying
impact services. Do not add its source to the public repository.

Docker builds require the forecast-core checkout in the build context. CI checks
out its private repository before building API. Commit and push private backend
changes before deploying public code that requires them.

## Configuration

Set `ARGUS_WORKDIR` when running an installed package outside the application
checkout. Otherwise configuration discovery searches the current directory and
its parents for `configs/project.yaml`; it does not depend on installation paths.

## Validation

Run the dependency guard without the private backend:

```bash
uv run --frozen pytest tests/test_architecture.py packages/forecast/tests
```

Integration tests also require API dependencies and the private checkout.
Private implementation tests are in `packages/forecast-core/tests`; those touching
API storage are integration tests and require the API source on `PYTHONPATH`.
Use the repository test runner, `scripts/test-python`, for the combined suite.
The root test environment also needs the API runtime dependencies, listed in
`apps/api/pyproject.toml`, including sentry-sdk, SQLAlchemy and psycopg. The API's
own `.venv` is used by the command-import test after the sync above.

Legacy experimental training and HUXt scripts are preserved in the private
backend's `legacy_scripts` directory. They are not production commands; its README
records the retired feature-schema dependency in the older speed scripts.
