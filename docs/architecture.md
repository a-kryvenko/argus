# Architecture

Clio → Prophet → Intelligence exchange versioned HTTP contracts. Each service
accesses only its own database. API reads all three; Web uses API.

## Services and packages

| Component | Owns |
| --- | --- |
| `apps/clio` | Collection, original files, observation history and receipts |
| `apps/prophet` | Prepared input snapshots, forecast releases and verification |
| `apps/intelligence` | Spacecraft, immutable assessments, retained input releases and journal |
| `apps/api` | Public HTTP, dashboard accounts, sessions and usage statistics |
| `apps/web` | UI |
| `packages/common` | Shared configuration and transport contracts |
| `packages/forecast` | Public inference and feature primitives |
| `packages/forecast-core` | Private forecasting backend |
| `packages/intelligence-core` | Private impact calculations and risk policy |

## Dependencies and configuration

| Consumer | Allowed packages |
| --- | --- |
| Clio, API | `common` |
| Prophet | `common`, `forecast`, `forecast_core.api` |
| Intelligence | `common`, `intelligence_core.api` |
| `forecast` | `common` |
| `forecast-core` | `common`, `forecast` |
| `intelligence-core` | `common` |

Applications never import another application's runtime. Private backends are
separate ignored Git checkouts; the public workspace resolves without them.
Training and research stay in private notebooks/backend repositories.

Configuration: `configs/`, rooted at `ARGUS_WORKDIR`. Command adapters load root
env files. Containers receive only their own DB credentials and necessary service
tokens. Observation, forecast and Intelligence tokens must differ.

## Runtime ownership

| Process | Data mounts |
| --- | --- |
| Clio HTTP/worker | `data/observations`, writable |
| Prophet worker | `data/prophet` and `data/demo`, writable; `data/models`, read-only |
| PROSWIN executor | Prophet cache, writable; models, read-only |
| API | Metrics and demo exports, read-only |
| Prophet HTTP, Intelligence HTTP/worker | None |

Readiness checks local resources. Upstream startup uses `service_started`;
unavailable inputs do not block reading stored results. Separate DB owners enforce
access within a shared PostgreSQL instance. Development source mounts and shared
infrastructure are not a hostile-code security boundary.

## Independent service development

- **Clio:** improve observation quality and provenance; publish observations and checksum-verified originals.
- **Prophet:** prepare calibrated features, freeze inputs, calculate and publish each product independently. Failed products retain their previous release. Calculation children never write to DB.
- **Intelligence:** evaluate supplied trajectories against published density/background and CME releases. All sampling, attention and freshness policy live in the private core.

Reads never trigger collection or forecast generation. Add optional fields within
contract v1; incompatible changes require a new version and migration path.
Model/policy changes must remain attributable in outputs.

New density assessments atomically retain exact input releases, deduplicated by
ID and verified by SHA256. Changed content under the same ID aborts publication.
Reads of retained inputs use only Intelligence's DB. Older assessments and the
experimental loss monitor have no retained inputs; reuse does not backfill them.
Replay also requires the matching private core and configuration.

Clio `as_of` is not a historical-availability guarantee for every observation.
Prophet's frozen prepared snapshot is its replay boundary. Capture timestamps
must not be presented as original provider publication times.

## Validation

```bash
./scripts/test-python -q
./scripts/test-domain-storage
pnpm docs:architecture:check
```

Python suites use each owner's locked environment and eight workers (`-n 0` for
sequential execution). Private suites require their checkouts. Domain-storage
checks require Docker and create a disposable PostgreSQL; local credentials are
not loaded. `TEST_DATABASE_ADMIN_DSN` may select an existing disposable server.

Checks cover imports/facades, manifests and lockfiles, foreign-domain SQL,
Compose ownership, migrations, cross-DB access and atomic publication.
Project-wide tests live in `tests/`; service/package tests live with their owner.

[Setup](../README_DEPLOY.md) · [Commands](commands.md) · [Diagrams](architecture/README.md)
