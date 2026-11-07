# Clio

Sources and policies: [project.yaml](../../../configs/project.yaml).

- `sources.live` / `sources.historical`: products in priority order; an empty list disables that mode.
- `retention_days`: file retention; `backfill_days`: historical recovery window, no longer than retention.
- `freshness_minutes`: maximum source observation age; `poll_minutes`: worker interval.
- `resolution_minutes` / `live_resolution_minutes`: historical/live sampling intervals. NRT FITS intervals must be multiples of its native hour.

Sampling intervals, polling and backfill windows are required; file policies also require explicit retention.

SDO products use `aia_094_nrt`, `aia_094_browse`, etc. NRT recovery uses the same dated NRT source; browse has no history. Missing source timestamps produce unknown freshness.
Originals and derivatives live in S3; PostgreSQL stores file references and observation metadata.
`GET /internal/v1/files?observation=aia_171_nrt&start=…&end=…` returns stored originals for a file observation.
Files are downloaded to `tmp`, uploaded to S3, recorded in `file_observation`, then removed from `tmp`.

`clio` serves authenticated HTTP; `clio-worker` runs collection, backfill, normalization and cleanup.
Health: `/health/live`, `/health/ready`. Logs: `./argus logs clio`.

[Setup](../../../README_DEPLOY.md#local-development) · [Commands](../../../docs/commands.md#clio)
