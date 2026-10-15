# Clio

Collects, archives and serves observations. No model features or calibration.

## Dataflow

- **Input:** External observations, solar images and event catalogs.
- **Output:** Stored observations, checksum-verified originals and collection status over HTTP to Prophet and API.

Event inputs contain catalog observations and external ENLIL predictions with receipt hashes. Prophet owns arrival forecasting and version selection.

[DATA_FLOWS.md](DATA_FLOWS.md).

## Observations

[project.yaml](../../configs/project.yaml) defines sources, priorities, polling and backfill.
Polling frequency differs from provider publication cadence.
Backfill preserves existing values/first receipts; file recovery requires matching SHA256.
[Model input requirements](../../docs/observation-requirements.md).

## Operation

```bash
./argus clio collect
./argus clio backfill
./argus clio normalize
./argus clio worker
```

Authenticated status: `/internal/v1/observations/status` and `/monitoring`.
`/health/live` checks HTTP; `/health/ready` checks storage;
`clio check-health worker` checks the worker heartbeat.

[Setup](../../README_DEPLOY.md#local-development) · [Commands](../../docs/commands.md#clio)
