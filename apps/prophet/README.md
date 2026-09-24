# Prophet

Prophet owns forecast runs, saved inputs, artifacts, releases and hourly
slots. Production uses one image for `prophet` (worker) and `prophet-api` (HTTP).
It reads Clio over HTTP and never accesses Clio tables. See
[setup](../../README_DEPLOY.md#local-development) and [commands](../../docs/commands.md).

## Service layout

`src/argus_prophet/services/` groups forecast operations by responsibility:

- `generation/`: product catalog, model loading, calculation and the product cycle.
  `cycle.py` reads one input snapshot and records each selected product's outcome;
  `calculation.py` computes and serializes an individual product.
- `density/`: atmospheric density input preparation and calculation.
- `releases/`: publication, release reads and status diagnostics.
- `inputs.py`: the Clio HTTP input client.
- `runs.py`: run recording, input snapshots, artifacts and provenance.

`cli.py` dispatches commands, `worker.py` owns scheduling and the generation lock,
and `main.py` serves HTTP. Package initializers do not re-export service modules;
model backends remain lazily imported when their product is calculated.

## Configuration and inputs

Configure `PROPHET_DB_*`, `OBSERVATIONS_URL`, `OBSERVATIONS_SERVICE_TOKEN` and
`FORECASTS_SERVICE_TOKEN`. Generation requires the private forecast-core checkout
and configured models. Clio's [input contract](../clio/README.md#configuration-and-interfaces)
is read once per full generation, including density, with 180s read and 10s connect
timeouts. Prophet saves the exact response, dependency/model hashes and compressed
CSV results. Model binaries are not archived; retain them separately for replay.

## Generation

`services/generation/products.py` defines supported products, their artifacts, backend adapter paths
and calculation modes. Solar-wind adapters come from the public `forecast.api`;
other model adapters come from `forecast_core.api`. Classes are imported only
when a selected product is calculated. There is no separate model enumeration.
`services.generation.calculation.calculate` is the shared runner for scheduled execution and local
development. It requires a saved input snapshot and run recorder; product-specific
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
loaded. `forecast.api.calculate_forecast` accepts an already loaded service,
observations and explicit issue time; it returns a `ForecastResult` containing a
DataFrame and model metadata without configuration, database or filesystem access.
`services.density.forecast.calculate_density` returns the same result structure.

The generation runner serializes each result once to UTF-8 CSV in memory and passes
bytes to `RunRecorder.store`, which compresses and hashes them for PostgreSQL.
Generation creates no output files, temporary CSVs or file archives. Filesystem
export and its retry/tracking subsystem have been removed. There is no
`ForecastDirector`, file-reading fallback or storage callback in calculation code.

## Publication and HTTP

Each product has its own run: `running`, `succeeded`, `failed` or `interrupted`.
Historical `partial` batch runs remain readable. A product publishes only when all
its required artifacts share one run/issue time. An incomplete product cannot be
marked successful. Product completion, publication and the hourly slot's aggregate
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

The worker generates the latest due hour at `:10 UTC` and retries every 60 seconds.
Only products without a successful scheduled attempt in that slot are retried.
A new cycle reads a fresh snapshot for these products. Once every supported product
succeeds, the slot is `succeeded`; otherwise it is `partial` (some successes) or
`failed` (none). Slot `attempts` counts product attempts for new runs. Historical
batch slot counters are retained as recorded.

On restart, committed products are not recalculated. Historical batch releases
also count as product successes. Manual runs do not consume scheduled work. A new
due hour starts a full cycle and abandons retries for older hours; missed hours are
not replayed. Slots are scheduling records, not issue times or release IDs.

A PostgreSQL session advisory lock serializes generation. All writes reuse its
unpooled connection in short transactions; no transaction spans model execution.
Use direct or session-pooled PostgreSQL, not transaction pooling. After session
loss an old writer cannot reconnect silently. The next lock owner marks abandoned
runs/slots interrupted. SIGTERM permits the active cycle to finish within Compose's
grace period. There is no automatic run/release retention.

Migration `20260918_prophet_no_exports` removes `forecast_export` and
`forecast_artifact.csv_written_at`, retaining all releases and compressed artifact
bytes. Apply migrations before starting the updated worker and read service.

Advanced recovery uses the internal adapter `scripts/dev/run` locally or
`scripts/prod/run` on the server: `prophet runs`, `prophet show-run <uuid>`,
`prophet slots`. Preserve database, image and model
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


## Wind model rollout

Both wind-speed outputs now resolve to `argus-plasma-speed-aia-ridge-v1.joblib`.
Deploy that artifact and the matching registry together. Clio must first migrate
`20260923_aia_snapshot` and start hourly AIA collection. Its optional `aia_frames`
input is passed alongside observed speed history; absence falls back to DLinear.
The serialized artifact includes all model dependencies and uncertainty samples.
Historical2025 metrics are under `data/metrics/plasma`; nominal intervals
are empirically undercovered (about73% at96h for nominal80%), not guaranteed coverage.
