# Solar event forecasts

The separate public product is `GET /public/forecasts/solar-wind-events`; its page is `/products/solar-events`.
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

Run the private release workflow and verification before registering an artifact. Model bytes belong in
`data/models/`; the registry pins their SHA256. An absent or mismatched artifact disables only the speed outlook.
No database schema changes are needed for the speed outlook; it travels inside the existing versioned event report.
The DONKI collection migration is a separate prerequisite on installations without the event receipt table.
