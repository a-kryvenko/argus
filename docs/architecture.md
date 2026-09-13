# Package and application boundaries

| Component | Responsibility | Project dependencies |
| --- | --- | --- |
| common | Configuration, serialization, shared contracts | None |
| packages/clio | Provider observation fetch/parse | common |
| forecast | Forecast interfaces and CSV products | common; optional forecast_core.api |
| forecast-core | Private models, feature preparation and calibration | common, clio |
| apps/clio | Observation storage, collectors, schedules and read HTTP API | common, clio; private calibration through services/calibration.py |
| apps/prophet | Forecast execution and scheduling; observation reads through HTTP | common, forecast, forecast_core.api |
| apps/api | Public HTTP/authentication; own dashboard storage; Clio HTTP client | common, forecast |
| intelligence-core | Private impact calculations, not yet integrated | Shared contracts as needed |

Clio owns schema `clio` and its migrations under
`apps/clio/src/argus_clio/migrations`. API owns schema `api` and migrations under
`apps/api/alembic`. Separate runtime/migration roles enforce ownership; API reads
observations only through Clio contracts. Prophet has no SQL credentials yet.
See [domain storage and cutover](domain-storage.md) and
[extraction stages](service-extraction.md).

Provider parsing belongs to the public clio library. Private calibration and
legacy model-input normalization stay in forecast-core and are called through a
lazy Clio ingestion adapter; the Clio read path does not import the backend.
Atmospheric density is a forecast product; satellite-specific impact belongs to
intelligence-core. Public code must not import private implementation modules.
Backend source, training and implementation-specific tests remain private.

## Checkouts and installation

The public Python workspace contains common, clio and forecast and resolves
without private repositories. API also installs without private code.
`packages/forecast-core` is a separate ignored Git checkout used by Clio and
Prophet. Keep its source/version consistent with both application lockfiles.

```bash
uv sync --project apps/api --frozen
uv sync --project apps/clio --frozen
uv sync --project apps/prophet --frozen
```

After a backend dependency change, update the Clio/Prophet lockfiles before
syncing. Docker builds for these two apps require the private checkout. CI checks
it out before building; commit/push private changes before deploying dependent
public code. API's Docker image contains no private backend or provider library.

The ignored `private/intelligence-core` directory is not yet used by applications.
Move it to a dedicated private repository before deploying impact calculations;
do not publish its source in this repository.

## Configuration and commands

Set `ARGUS_WORKDIR` when running outside a checkout. Otherwise configuration
searches the current directory and parents for `configs/project.yaml`, independent
of package installation paths. Read [Clio setup](../apps/clio/README.md) for domain
credentials, bootstrap and migration commands; read
[Prophet setup](../apps/prophet/README.md) for forecast commands.

## Validation

```bash
uv run --frozen pytest tests/test_architecture.py packages/forecast/tests
./scripts/test-python -q
```

The combined suite includes Clio, API, Prophet and private backend tests. Private
implementation tests remain in `packages/forecast-core/tests`. The runner sets
application/package import paths; installed-environment checks also use the API,
Clio and Prophet virtual environments after syncing them. PostgreSQL integration
requires an explicitly configured disposable server as described in
[domain storage](domain-storage.md); public CI provides one automatically.

Legacy experimental HUXt/training scripts remain in the private backend's
`legacy_scripts` directory and are not production entry points.
