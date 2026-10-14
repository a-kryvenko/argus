# Solar event forecasts

The separate public product is `GET /public/forecasts/solar-wind-events`; its page is `/products/solar-events`.
The main `GET /public/forecasts/solar-wind-speed` product also includes `cme_speed` from its own
saved release. The speed chart and inspector display the nominal, early and late CME scenarios
alongside background quantiles. Background threshold probabilities retain their original meaning.
Prophet stores each release and its causal input snapshot in the normal forecast ledger.
Clio stores versioned DONKI/ENLIL responses with receipt times and content hashes.

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
