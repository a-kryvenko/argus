# Clio

Collects, archives and serves observations. No model features or calibration.

## Dataflow

- **Input:** External observations, solar images and event catalogs.
- **Output:** Stored observations, checksum-verified originals and collection status over HTTP.

Event inputs contain catalog observations and external ENLIL predictions with receipt hashes.

[DATA_FLOWS.md](../DATA_FLOWS.md).

## Observations

[project.yaml](../../../configs/project.yaml) defines sources, priorities and backfill.
Polling frequency differs from provider publication cadence.
Backfill preserves existing values/first receipts; file recovery requires matching SHA256.
[Model input requirements](../../../docs/observation-requirements.md).

## Operation

```bash
./argus clio collect
./argus clio backfill
./argus clio normalize
```

Authenticated status: `/internal/v1/observations/status` and `/monitoring`.
`/health/live` checks HTTP; `/health/ready` checks storage.
Cron uses the host timezone and `flock`; logs: `log/clio/`.
The host needs Docker Compose, cron, flock and Python 3.
Deployment installs [cronjobs.txt](../cronjobs.txt) with the other services and deploy jobs.

## File storage rollout

Release separately, in order:

1. Previous release: add `file_observation`, read both schemas, write both, backfill history.
2. Current: read and write only file storage. Cutover migration verifies archive coverage and checksums; legacy tables remain.
3. After every process runs stage 2: drop legacy tables and model definitions. This removes rollback support for stage 1 and earlier.

Do not ship later stages as pending migrations in the current release.
Before deploying stage 2, finish backfill and run `./argus clio backfill-files --check` on stage 1. Pre-stage-1 writers must have finished.

[Setup](../../../README_DEPLOY.md#local-development) · [Commands](../../../docs/commands.md#clio)
