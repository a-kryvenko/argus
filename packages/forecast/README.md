# Public solar-wind forecasting

This package contains the public solar-wind inference implementation used by
Argus Sunwatch. It runs without the private `forecast_core` package, a database,
service credentials, network access or project configuration.

## Scope

- Solar-wind speed: q10/q50/q90 and threshold-exceedance probabilities.
- Solar-wind density: q10/q50/q90.
- Observation feature construction, horizon buckets, smooth overlapping quantile
  blending and application of previously fitted calibration parameters.
- Shared model interfaces and in-memory results, also reused by private products.

The stable entry point is `forecast.api`. Models implement `predict` or
`predict_proba`; the package does not train or load production model files.
Geomagnetic, magnetic-field, radiation and atmospheric-density products belong
to the private backend. `forecast` never imports that backend.

## Reproduce the public example

From the repository root, in a Python 3.12+ environment:

```bash
python -m pip install -e packages/common -e packages/forecast pytest
python -m forecast.demo
python -m pytest packages/forecast/tests
```

The example uses eight synthetic hourly observations and constant toy estimators.
It exercises all three public inference adapters and emits six hourly rows per
artifact. Inputs and issue time are fixed, so repeated executions produce the
same output. It needs neither a private checkout nor production model weights.
The output demonstrates the calculation pipeline, not predictive accuracy.

Production model runtimes can be installed with the optional `models` extra:
`python -m pip install -e 'packages/forecast[models]'`. Trained bundles and their
validation data must be supplied separately.

## Calculation contract

```python
from forecast.api import SWSpeedFS, calculate_forecast

result = calculate_forecast(
    SWSpeedFS(model_bundle),
    observations,
    issue_time=issue_time,
    model_info={"model": "my-speed-model", "sha256": model_hash},
)
# result.name, result.frame, result.model_info
```

Observations use `common.schemas.observation.Observation`. Feature construction
expects more than six chronologically ordered hourly rows, including `bx`, `by`,
`bz`, `v`, `n`, `t`, `kp`, `ap`, `dst` and `f10_7`. Geomagnetic and solar-index
observations are input features; their forecasting models are not part of this
public package. Missing-input policy belongs to the caller.

Bundles provide `models`, `feature_columns`, `lead_hours` and `buckets`.
Threshold models use `(threshold, bucket_label)` keys. Quantile models use
`(bucket_label, "q10"/"q50"/"q90")` keys, or numeric `(start, end)` bucket tuples
for overlapping horizons. Overlapping quantile bundles may contain a fitted
`calibration` DataFrame with `lead_hours`, `median_bias`, `lower_scale` and
`upper_scale`. Training and calibration fitting are separate from inference.

`issue_time` must be timezone-aware. Valid times advance in hourly steps from
that issue time rounded down to the hour. Results include the explicit issue
time, lead hours and valid times. Feature construction and forecast computation
perform no storage operations.

## Verification and research reporting

The public CI runs the entire test suite of this package without private code.
It checks feature values, deterministic output, quantile blending/calibration,
threshold probabilities and time alignment. These are software correctness
checks; scientific performance requires evaluation on real held-out data.

For a reproducible performance report, record the code revision, model hash,
input sources and time range, training/test split, baselines, metrics per forecast
horizon and known limitations. No performance claims or operational model weights
are included in the synthetic example.
