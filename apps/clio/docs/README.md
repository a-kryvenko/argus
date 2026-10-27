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

1. Current: add `file_observation`; prefer file receipts, include legacy-only records, write both schemas. Cron copies missing history in batches.
2. After old writers finish and `./argus clio backfill-files --check` succeeds: read and write only file storage; remove the backfill cron job.
3. After every process runs stage 2: drop legacy tables and compatibility code. This removes rollback support for stage 1 and earlier.

Do not ship later stages as pending migrations in the current release.
Retry a batch manually: `./argus clio backfill-files --batch-size 32`.

[Setup](../../../README_DEPLOY.md#local-development) · [Commands](../../../docs/commands.md#clio)
