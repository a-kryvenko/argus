# Intelligence integration stub

This independent CLI validates one complete published Prophet release using its
HTTP contract. Success means integration and contract validation succeeded; it
makes no statement about freshness, model readiness or satellite risk.

```bash
uv sync --project apps/intelligence --frozen
./scripts/intelligence check
./scripts/intelligence check dst --release-id <uuid>
```

Local use requires `FORECASTS_URL` and `FORECASTS_SERVICE_TOKEN`. On production,
Compose supplies the internal URL and existing forecast service token. No new
manual env values, database credentials, migrations or private checkout are needed.

```bash
/var/www/bin/argus intelligence check
/var/www/bin/argus intelligence check dst --release-id <uuid>
```

The default product is solar-wind-speed. JSON output identifies the actual
`release_id` and `run_id` checked, reports `mode: stub` and `risk_assessment: null`.
The latest endpoint returns a complete release in one response; an explicit ID
supports repeatable checks. Missing releases, failed HTTP requests, oversized
responses or invalid artifacts exit with code 1 and never report success.

No files or database records are written. Save stdout externally if an audit
record is needed. The tools-profile container runs only when requested, with
no mounted data, exposed ports or access to the database network. Scheduling and
actual risk calculations are separate next steps.
