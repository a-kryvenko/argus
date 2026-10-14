# Solar event forecasts

The separate public product is `GET /public/forecasts/solar-wind-events`; its page is `/products/solar-events`.
The main `GET /public/forecasts/solar-wind-speed` product also includes `cme_speed` from its own
saved release. The speed chart and inspector display the nominal, early and late CME scenarios
alongside background quantiles. Background threshold probabilities retain their original meaning.
Prophet stores each release and its causal input snapshot in the normal forecast ledger.
Clio stores versioned DONKI/ENLIL responses with receipt times and content hashes.
Clio also normalizes external arrivals, selects causal CME analyses and exposes their
initial speeds and simulation identities. Prophet and forecast-core consume these normalized
events; they no longer interpret DONKI arrival responses or CME speed analyses.
Deploy/migrate Clio before updating Prophet. Older snapshots without normalized events are
rejected explicitly; the main speed product retains its background with an unavailable CME outlook.

Clio's independent `collect-enlil` job collects compact NOAA numerical L1 profiles from
`https://noaa-wsa-enlil-pds.s3.amazonaws.com/`. It runs with the enabled solar-event collector,
even when NASA DONKI collection fails. It stores original NetCDF bytes, hashes, provider
modification/model creation times and actual local receipt times in `enlil_profile_receipt`.
The new `20261014_enlil_profile` migration is required. Repeated unchanged objects are skipped.
The internal event-input contract exposes the newest causally available profile and source status.
NOAA profiles remain distinct from NASA DONKI simulations and are not automatically linked to
NASA arrivals or used to replace the approved speed model. Density is proton-equivalent mass
density in cm^-3; velocity is the vector magnitude in km/s. Only profiles labelled L1 are accepted.
Discovery checks the last three days; no recent compact data is reported as a source failure,
while prior receipts remain available for replay. An expired profile is marked stale.

The current release publishes CME arrivals and an optional CME-conditioned solar-wind speed outlook:
background, nominal impact, early and late arrival scenarios, and a scenario envelope in km/s.
These curves are conditional scenarios, not quantiles or calibrated impact probabilities.
Missing source/model inputs are reported as unavailable; they do not imply a quiet Sun.
Unsupported multi-CME speed estimates remain visible as exclusions while arrival predictions are retained.
HSS and DBEM forecasts are disabled in this release. L1 compression candidates do not automatically confirm CME attribution.

Training and validation implementations, protocols and detailed results live in the private
`notebooks/workflows/solar_events/` checkout. Real-time algorithms live in the private
`packages/forecast-core/src/forecast_core/solar_events/` checkout. Public applications own collection,
scheduling, input contracts, artifact serialization, API and rendering. Research commands run directly from the private notebooks checkout.

The approved split is training through2024, calibration/selection or fine-tuning in2025, evaluation only in2026.
The old numerical-profile experiments `enlil_profiles_archive_v1` and
`enlil_profiles_model_time_v1` are ineligible: their initial correction fit used2025 data.
Their artifacts are retained for audit and marked with eligibility reports; re-running the fitted
workflow now requires pre2025 training profiles and labels. The replacement private workflow
`audit_raw_enlil_profiles.py` evaluates raw numerical profiles without any fitting or parameter
selection. The available local compact-profile manifest starts in August2025, so it cannot support
initial training under the approved protocol. Archive modification times provide a replay proxy,
not verified original publication receipts. Window peak scores include ambient/HSS structures and
overlapping windows; they must not be described as independent CME detection scores.
An inspected NOAA archive day in2024 contains full3D gzip tarballs of roughly31GB per run;
a bounded2MB prefix inspection did not reach a compact L1 profile. Older numerical data therefore
exist, but efficient extraction or another compact historical source is needed before initial fitting.
The subsequent complete2024 inventory found584 full3D archives (18.26TB total) and one separately
accessible original L1 profile, run56401 on May11. The latter was downloaded and validated:
model creation in2024,13662 L1 points, and11 input CMEs. It was uploaded to the compact archive
in July2026, so original publication is not verified. This single complex case does not establish
representative training coverage. The private `probe_enlil_history_sources.py` workflow records
bounded HTTP receipts, hashes, archive-prefix inspections, compact samples and acquisition costs.
NASA ISWA's historical numerical-product directories are another candidate, but direct retrieval
timed out during this audit. No2025/2026 data were fitted and no production model was changed.
The release builder rejects late labels and cross-partition events, reproduces the prior peak/shape selection,
and freezes model hashes before evaluation. Historical results have already been inspected;2026 is not used for fitting.

The approved CME artifact ships inside the private forecast-core wheel and Prophet image, so ordinary
deployment does not require manually copying a model to the host. Release planning verifies the bundled
artifact against the registry SHA256 before building images. Changing the model rebuilds Prophet.
An optional `data/models/<model>.joblib` override is also checked against the registry; a mismatch disables
the speed outlook. The normal bundled artifact is used when the host override is absent.
CME speed scenarios are stored inside the same `plasma_speed_quantile` artifact as their baseline;
release validation checks their time grid and background values. The API never joins a newer event
release onto an older speed forecast. Missing event inputs leave the background forecast available
and mark the CME outlook unavailable.
No database schema changes are needed for the speed outlook; it travels inside the existing versioned event report.
The DONKI collection migration is a separate prerequisite on installations without the event receipt table.
