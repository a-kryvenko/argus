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

## File storage

`file_observation` stores receipt times, paths and SHA256; gzip originals and receipt metadata live in `data/observations/`.
Before deploying the legacy-table cleanup, every process must run the file-only release (`20261027_file_cutover`).
Rollback to the earlier dual-storage release requires restoring a backup.

[Setup](../../../README_DEPLOY.md#local-development) · [Commands](../../../docs/commands.md#clio)
