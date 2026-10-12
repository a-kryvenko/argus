# Forecast contracts

Public forecast interfaces, input schemas, calculation contracts and evaluation helpers for Argus Sunwatch.

Model implementations, feature engineering, blending, calibration and training live in the separately installed private `forecast-core` package. `forecast.api` loads model adapters lazily; importing contracts does not require the private backend. Requesting a model adapter without it raises an explicit installation error.

Existing model import paths remain thin compatibility interfaces, including paths referenced by saved artifacts. They contain no fitted weights or modeling recipes. Run model and compatibility tests in the private backend; public tests verify contracts and backend isolation.

Research command namespaces under `scripts/revalidation` and `scripts/training` forward to the private source checkout. Notebooks remain in the private `notebooks/` checkout. Deploy the public interface and private implementation revisions together.
