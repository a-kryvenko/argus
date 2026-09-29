# Prophet

Prophet owns forecast runs, saved inputs, artifacts, releases and per-product schedule
slots. Production uses one image for `prophet` (worker) and `prophet-api` (HTTP).
It reads Clio over HTTP and never accesses Clio tables. See
[setup](../../README_DEPLOY.md#local-development) and [commands](../../docs/commands.md).

Published-release verification: `./argus prophet verify [product] [--days 7]` and
`./argus prophet verification-report [product]`. Apply migration
`20260926_prophet_verification` first. Verification uses original Clio measurements
over HTTP and stores scored pairs/metrics in Prophet, separately from website
model metrics. See the [two forecast workflows](../../docs/forecast-workflows.md).

## Service layout

`src/argus_prophet/services/` groups forecast operations by responsibility:

- `generation/`: product catalog, model loading, calculation and the product cycle.
  `cycle.py` reads one input snapshot and records each selected product's outcome;
  `calculation.py` computes and serializes an individual product.
- `releases/`: publication, release reads and status diagnostics.
- `inputs.py`: the Clio HTTP input client.
- `runs.py`: run recording, input snapshots, artifacts and provenance.

`cli.py` dispatches commands, `commands/` handles operational tasks, and
`main.py` serves HTTP. `scheduling/jobs.py` owns slots and writer locks;
`scheduling/execution.py` supervises calculation processes; `worker.py` dispatches
generation and periodic verification. Package initializers do not re-export service modules;
model backends remain lazily imported when their product is calculated.

## Configuration and inputs

Configure `PROPHET_DB_*`, `OBSERVATIONS_URL`, `OBSERVATIONS_SERVICE_TOKEN` and
`FORECASTS_SERVICE_TOKEN`. Generation requires the private forecast-core checkout
and configured models. Clio's [input contract](../clio/README.md#forecast-inputs)
is shared by products dispatched together (once per manual cycle), including density, with 180s read and 10s connect
timeouts. Prophet saves the exact response, dependency/model hashes and compressed
CSV results. Model binaries are not archived; retain them separately for replay.

## Generation

`services/generation/products.py` defines supported products, their artifacts, backend adapter paths
and whether they load a model bundle. Solar-wind adapters come from the public `forecast.api`;
other model adapters come from `forecast_core.api`. Classes are imported only
when a selected product is calculated. There is no separate model enumeration.
`services.generation.calculation.calculate_product` is the shared calculation entry point for scheduled
execution and local development. The coordinator saves the input snapshot before dispatch; product-specific
launch scripts and unrecorded CSV generation paths have been removed from Prophet. Public release contracts remain
independent and tests check that the generation catalog matches them.

Every product in a cycle receives the same snapshot and `issue_time`: the snapshot's
`as_of` floored to the UTC hour. ML output records this explicit issue time, not the
last observation timestamp. Input age remains available separately in diagnostics.
Model bundles still determine forecast horizons and model configuration remains in
`configs/models_registry.yaml`. Generation does not train models.

Use `./argus prophet refresh [product]` for local development. Only canonical
product names and `all` are accepted. Historical short names are recognized only
when reading old run records. Selecting Dst or solar-wind density computes only
that product.
The public HTTP release format is unchanged.

## Calculation and storage boundary

`services.generation.models.load_model` reads configured model bundles and fingerprints the exact bytes
loaded. Every adapter implements `forecast_snapshot(inputs, issue_time=...)`;
`forecast.api.calculate_snapshot` returns a `ForecastResult`. AIA, DLinear, GONG
and hourly IMF input preparation belongs to the respective adapter. The Prophet
runner has no branches inspecting private model fields. Atmospheric density uses
`forecast_core.api.AtmosphericDensityForecastService` directly: snapshot preparation,
physical drivers and numerical calculation all belong to the private backend.

Each product runs in a fresh spawned process. The parent supplies the saved
snapshot, model directory and registry explicitly; the child reads model bundles,
calculates and returns serialized artifacts. It never receives the database
connection. The parent retains the generation lock, stores artifacts and publishes.
An exception, crash or timeout fails only that product; every child is reaped
before its result is handled; all active children exit before the lock is released. Private `forecast_core.api`
exports are lazy, so loading Dst does not import GONG or JB2008.

The generation runner serializes each result once to UTF-8 CSV in memory and passes
bytes to `RunRecorder.store`, which compresses and hashes them for PostgreSQL.
Generation creates no output files, temporary CSVs or file archives. Filesystem
export and its retry/tracking subsystem have been removed. There is no
`ForecastDirector`, file-reading fallback or storage callback in calculation code.

## Publication and HTTP

Each product has its own run: `running`, `succeeded`, `failed` or `interrupted`.
Historical `partial` batch runs remain readable. A product publishes only when all
its required artifacts share one run/issue time. An incomplete product cannot be
marked successful. Product completion, publication and its own schedule slot's
status commit in one transaction.

The cycle continues after model, input-validation or publication errors for an
individual product. Successful products publish immediately; failed products keep
their previous release. Missing density inputs follow the same failure policy as
any other product. A failed cycle returns a nonzero exit status after attempting
all selected products. Database/lock loss that prevents recording a failure stops
the cycle, and the next writer recovers abandoned work.

Clio is read once per cycle, and the identical response is saved in each product's
run; dependency/source provenance is collected once. If Clio fails, each selected
product gets a failed attempt and no calculation runs. Status reads distinguish
that product's attempt from unrelated products' successes.

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

The worker dispatches each product independently. Default schedules remain hourly
at `:10 UTC`; `prophet.schedules.<product>` sets `every_minutes` and
`offset_minutes`. For example, `dst: {every_minutes: 5, offset_minutes: 0}` runs
Dst every five minutes. Offsets must be smaller than the interval. The worker
checks eligibility and retries failures every 60 seconds. At most
`prophet.max_parallel_products` calculations run concurrently (default 2), and
the same product never overlaps itself. Products dispatched together share an
input snapshot; later dispatches fetch fresh inputs without waiting for a full cycle.

Migration `20260929_prophet_product_slots` changes slot identity to `(product, slot)`
and permits subhour slots. Historical aggregate slots and their run references
are retained as `product=all`. New slot counters count attempts of that product.
A committed success is not repeated after restart or clock rollback; historical
batch releases also count as successes. Manual runs do not consume scheduled work.
Each product uses only its latest due slot; missed intervals are not replayed.
Slots describe execution cadence. Existing models still use hourly issue times
and output grids, so more frequent runs may publish updated forecasts for the same
issue hour. Changing model resolution requires a separate model/contract change.

A PostgreSQL session advisory lock serializes generation coordinators. The owning
thread saves inputs, artifacts and releases through its unpooled connection in
short transactions; no transaction spans model execution. Temporary processes
fetch inputs or calculate products; they never inherit this connection. Use direct
or session-pooled PostgreSQL, not transaction pooling. After session loss an old
writer cannot reconnect silently. The next lock owner marks abandoned runs/slots
interrupted. SIGTERM/SIGINT stops new dispatch and drains active calculations,
with a default 570s deadline inside Compose's 10-minute grace. At the deadline,
children are terminated and their attempts recorded as interrupted. Each input
fetch or calculation has its own default 540s limit, configurable in
`prophet.calculation_timeout_seconds`.

Configured verification has an independent process and writer lock, so it can
run while forecasts are calculated. The default schedule in `configs/project.yaml`
is every six hours, scoring seven days of published releases. Completion is saved
only after successful scoring in `prophet.scheduled_job` (migration
`20260929_prophet_jobs`). A failed pass retries after 60 seconds; manual verification
never advances the scheduled marker. The scoring budget defaults to 300s, with a
process limit 30s longer. Partial evidence may be saved but does not mark the pass
complete. Cleanup acquires both generation and verification locks.

`./argus prophet cleanup --days 90` previews up to 1000 old completed runs;
`--apply` deletes that batch under both writer locks. Current releases, the latest
attempt of every product, active runs, and all scheduling slots are retained.
Associated historical releases, artifacts and verification evidence are deleted
in one transaction. The minimum retention is 26 days. No automatic deletion runs.

Migration `20260918_prophet_no_exports` removes `forecast_export` and
`forecast_artifact.csv_written_at`, retaining all releases and compressed artifact
bytes. Apply migrations before starting the updated worker and read service.

Advanced recovery uses `./argus prophet runs`, `./argus prophet show-run <uuid>`
and `./argus prophet slots` in both environments. Preserve database, image and model
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
Status remains diagnostic. Generation separately enforces explicit policies in
`prophet.inputs.<product>`: `min_normalized_points`, `max_normalized_age_hours`,
and `max_gong_age_hours`. The configured HMF policy requires GONG within three
hours; other thresholds remain unset rather than implying fresh sensors from
filled normalized rows. Bundles retain their own history and coverage checks.
Policy failure records a failed attempt and keeps the previous release. Operational
settings are saved in run provenance. These checks do not establish forecast skill.


## Wind model rollout

Both wind-speed outputs now resolve to `argus-plasma-speed-aia-ridge-v1.joblib`.
Deploy that artifact and the matching registry together. Clio must first migrate
`20260923_aia_snapshot` and start hourly AIA collection. Its optional `aia_frames`
input is passed alongside observed speed history; absence falls back to DLinear.
The serialized artifact includes all model dependencies and uncertainty samples.
Historical2025 metrics are under `data/metrics/plasma`; nominal intervals
are empirically undercovered (about73% at96h for nominal80%), not guaranteed coverage.

When DLinear lacks observed speed history, generation reports the issue time,
required UTC windows, missing hourly intervals and the model's forward-fill limit.
The final generation error includes each failed product's reason. Normalized
observations do not substitute for missing observed speed history.

## GONG and backend rollout

This version requires `forecast>=0.3.0` and `forecast-core>=0.5.0`. Publish the
private revision before deploying dependent public code; both Clio and Prophet
lockfiles must accompany it. Apply Clio migration `20260929_gong_snapshot` and
Prophet migrations through `20260929_prophet_product_slots` before starting their updated services.

Run `./argus clio collect gong` to populate the new input, then
`./argus prophet refresh hmf`. Clio stores the exact original, checksum, first
receipt, source and versioned private features. HTTP reads return stored features
without loading the private backend. Prophet saves them in each input snapshot.
The model no longer reads `paths.live_gong`; absence of GONG is a visible product
failure, and historical snapshots without this field cannot replay southward IMF.
No model bundle paths or serialized model classes were renamed.
