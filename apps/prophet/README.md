# Prophet

Prophet owns forecast runs, saved inputs, artifacts, releases, exports and hourly
slots. Production uses one image for `prophet` (worker) and `prophet-api` (HTTP).
It reads Clio over HTTP and never accesses Clio tables. See
[setup](../../README_DEPLOY.md#local-development) and [commands](../../docs/commands.md).

## Configuration and inputs

Configure `PROPHET_DB_*`, `OBSERVATIONS_URL`, `OBSERVATIONS_SERVICE_TOKEN` and
`FORECASTS_SERVICE_TOKEN`. Generation requires the private forecast-core checkout
and configured models. Clio's [input contract](../clio/README.md#configuration-and-interfaces)
is read once per full generation, including density, with 180s read and 10s connect
timeouts. Prophet saves the exact response, dependency/model hashes and compressed
CSV results. Model binaries are not archived; retain them separately for replay.

## Publication and HTTP

Run states: `running`, `succeeded`, `partial`, `failed`, `interrupted`.
A product publishes only when all required artifacts share one run/issue time.
Run completion, publication and scheduled-slot completion commit together.
Failed runs retain previous releases; optional density skips allow other products
to advance. Manual runs do not advance scheduled slots.

Bearer-authenticated routes under `/internal/v1/forecasts/{product}`:

- `/latest`: current complete release.
- `/releases/{release_id}`: historical release.
- `/status`: release age, latest attempt and saved input diagnostics.

Contracts live in `common.schemas.forecast_release` and `forecast_status`.
Missing current releases/storage failures return 503; unknown products/historical
releases 404; invalid tokens 401. Status for a known product with no release returns
200 with `freshness=unavailable`. Reads share a repeatable-read snapshot; payloads
are limited to 32 MiB per artifact and 64 MiB per response. API has no CSV fallback.
The solar-radiation contract exists but no supported generator produces it.
Geomagnetic generation includes both Kp and Ap.

## Scheduling and recovery

The worker generates the latest due hour at `:10 UTC`, retries every 60 seconds
and does not replay missed hours. Partial runs complete their slot; export failures
do not reopen it. Slots are scheduling records, not forecast issue times or release IDs.

A PostgreSQL session advisory lock serializes generation and export. All writes
reuse its unpooled connection in short transactions; no transaction spans model
execution. Use direct or session-pooled PostgreSQL, not transaction pooling.
After session loss an old writer cannot reconnect silently. The next lock owner
marks abandoned runs/slots interrupted. SIGTERM permits the active job to finish
within Compose's grace period.

CSV exports retry independently of committed publication. Replacements are atomic
per file, not across a product or with database acknowledgement. Writers verify
their lock before replacement; superseded releases cannot overwrite newer exports.
There is no automatic run/release/archive retention.

Advanced recovery uses the internal adapter `scripts/dev/run` locally or
`scripts/prod/run` on the server: `prophet runs`, `prophet show-run <uuid>`,
`prophet slots`, `prophet export [--force]`. Preserve database, image and model
artifacts together when restoring; see [deployment](../../README_DEPLOY.md#recovery).

## Readiness diagnostics

Health endpoints report infrastructure readiness, not forecast fitness.
`./argus prophet status [product]` separates the published release from the latest
attempt. Release age uses model `issue_time`, not publication time. Freshness is
`unavailable`, `future_issue_time`, `unconfigured`, `stale` or `within_age_limit`;
the last confirms only an age check. Atmospheric density has a public maximum age
of 6h by default, configured in its registry entry. Other age limits are unconfigured.

Saved diagnostics describe input counts, timestamp range/age, gaps, duplicate,
naive/future timestamps and missing/nonfinite values. Density source metrics have
separate age/count evidence. Ages refer to the snapshot's `as_of`; older runs can
lack diagnostics. Status reads metadata without decompressing artifacts.

Normalization can fill old source values into recent rows, so
`source_freshness_known=false` and `thresholds_configured=false` remain explicit.
Diagnostics do not add generation gates. Per-product freshness/history requirements
and breach behavior still need definition; successful processing is not scientific
validation or proof of fresh sensors.
