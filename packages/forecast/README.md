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

The stable entry point is `forecast.api`. Existing adapters use models implementing
`predict` or `predict_proba`; the rotation DLinear adapter loads its versioned
inference bundle explicitly. The package does not train models.
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

## Rotation DLinear speed model

`forecast.api.RotationDLinearForecaster` applies a fitted speed-only model to a
DataFrame of `issue_time` / `lead_hours` requests using hourly `issue_time` / `v`
observations. It returns a copy with `rotation_v`, preserving row order and index.
Unavailable histories produce NaN; bounded causal forward fill and normalization
are defined by the artifact. Targets are never used for inference.

```python
from forecast.api import RotationDLinearForecaster

model = RotationDLinearForecaster.from_registry(
    workdir=workdir, registry=models_registry["models"]
)
result = model.add_rotation_v(requests, observations)
```

The `plasma_speed_dlinear` registry entry supplies `artifact_path`. The versioned
joblib bundle contains folded linear weights, bias and all preprocessing settings;
it needs NumPy/pandas and the optional `models` extra for joblib, not PyTorch.
Only load trusted joblib artifacts. Training remains separate, in
`notebooks/3_train_dlinear.ipynb`; the package performs inference only.

Quantile or threshold bundles that list `dlinear_v` in `feature_columns` must also
embed `feature_models["dlinear_v"] = {"bundle": <DLinear bundle>, "sha256": <snapshot hash>}`.
Forecast services compute this feature after expanding the forecast horizons,
using the embedded model rather than a mutable registry dependency. Pass unfilled
hourly speed as `speed_history` to `forecast` / `calculate_forecast` when the main
observation table has already been interpolated. Missing required history raises
an explicit error. Models that do not request `dlinear_v` keep the existing path.

Clio's forecast-input response includes `speed_observations` from raw measurements
over 60 days (hourly means, without interpolation); Prophet supplies this history
to dependent models. The configured v1 windows need 57 days plus a fill buffer.

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


## AIA193 wind and threshold model

The speed quantile and threshold registry entries share `argus-plasma-speed-aia-ridge-v1`.
It embeds the frozen speed-only DLinear, six fixed-window Ridge corrections, and
pre-2025 empirical error distributions. No research modules, FITS readers, PyTorch,
or LightGBM are required for this artifact at prediction time. Supply 60 days of
observed hourly `speed_history` and Clio's `aia_features` to `calculate_forecast`.
Missing/stale/unsupported AIA falls back to DLinear and its own error distribution.
The point forecast is unchanged; per-hour errors are centered to anchor q50 to it.
Q10/Q90 and P(V>=450/500/600) use this same distribution, with monotone thresholds.
Error calibration reuses 2024 validation folds, so it is not independent calibration.
On historical2025, nominal80% coverage at96h is about73%; thresholds are not claimed
uniformly superior to the old classifier. Metrics/provenance: `data/metrics/plasma/aia_ridge`.
