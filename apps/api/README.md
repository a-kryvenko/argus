# API

Both services use `/api/v1`. Public Go API schemas are at `/api/v1/docs`.

| Path | Purpose |
| --- | --- |
| `/observations` | Latest measurements and optional `trends=true` |
| `/observations/history` | History with provenance and aggregate coverage |
| `/observations/catalog` | Units, native cadence and sources |
| `/summary` | Published summary with S3 image URLs |
| `/forecasts/{target}` | Published forecasts, metrics and verification |
| `/risks/leo-drag` | [LEO drag calculator](../../docs/leo-drag.md) |
| `/dashboard/*` | Python API for accounts, monitoring and saved assessments |

Wind defaults to L1 time. Use `time_basis=near_earth&resolution=1h` for stored OMNI2 history. L1 raw retention and GOES backfill are six days. GOES history accepts up to seven days per request.

Dashboard cookie-authenticated writes and CORS use `DASHBOARD_ORIGINS`.

[Setup](../../README_DEPLOY.md#local-development) · [Commands](../../docs/commands.md)
