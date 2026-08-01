import numpy as np
import pandas as pd
import pytest

from forecast.inference._forecast_service import QuantileForecastService
from forecast.quantiles import (
    apply_quantile_calibration,
    learn_quantile_calibration,
    overlapping_bucket_weights,
)


class ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


class DummyQuantileForecastService(QuantileForecastService):
    target_name = "v"

    def _build_features(self, raw_observations_frame):
        return raw_observations_frame


def test_overlapping_weights_are_normalized_and_blend_smoothly() -> None:
    buckets = [(1, 4), (2, 6), (4, 16)]
    weights = overlapping_bucket_weights(np.arange(1, 17), buckets)

    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(weights[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(weights[15], [0.0, 0.0, 1.0])

    # At lead=3, the first two buckets cross-fade instead of hard-switching.
    np.testing.assert_allclose(weights[2], [0.6, 0.4, 0.0])


def test_quantile_service_blends_then_applies_calibration() -> None:
    buckets = [(1, 4), (2, 6)]
    models = {}
    for bucket, offset in [(buckets[0], 0.0), (buckets[1], 10.0)]:
        models[(bucket, "q10")] = ConstantModel(offset)
        models[(bucket, "q50")] = ConstantModel(offset + 10.0)
        models[(bucket, "q90")] = ConstantModel(offset + 20.0)

    calibration = pd.DataFrame({
        "lead_hours": np.arange(1, 7),
        "median_bias": 1.0,
        "lower_scale": 2.0,
        "upper_scale": 3.0,
    })
    service = DummyQuantileForecastService({
        "models": models,
        "buckets": buckets,
        "feature_columns": ["feature"],
        "lead_hours": 6,
        "calibration": calibration,
    })
    frame = pd.DataFrame({
        "lead_hours": np.arange(1, 7),
        "feature": 0.0,
    })

    result = service._build_forecast(frame, models, ["feature"])

    # Raw lead=3 blend is [4, 14, 24]; calibration shifts the median by one
    # and applies asymmetric scales around it.
    assert result.loc[2, "v_q10"] == pytest.approx(-5.0)
    assert result.loc[2, "v_q50"] == pytest.approx(15.0)
    assert result.loc[2, "v_q90"] == pytest.approx(45.0)


def test_learned_calibration_corrects_median_and_both_tails() -> None:
    targets = np.tile([-2.0, -1.0, 0.0, 1.0, 2.0], 2)
    calibration_frame = pd.DataFrame({
        "lead_hours": np.repeat([1, 2], 5),
        "target": targets,
        "q10": -1.0,
        "q50": 0.0,
        "q90": 1.0,
    })

    calibration = learn_quantile_calibration(
        frame=calibration_frame,
        target_column="target",
        prediction_columns=["q10", "q50", "q90"],
        smoothing_window=1,
    )
    predictions = calibration_frame[["q10", "q50", "q90"]]
    calibrated = apply_quantile_calibration(
        predictions=predictions,
        lead_hours=calibration_frame["lead_hours"],
        calibration=calibration,
    )

    np.testing.assert_allclose(calibrated["q10"], -2.0)
    np.testing.assert_allclose(calibrated["q50"], 0.0)
    np.testing.assert_allclose(calibrated["q90"], 2.0)


def test_calibration_can_preserve_the_raw_median() -> None:
    frame = pd.DataFrame({
        "lead_hours": 1,
        "target": [8.0, 9.0, 10.0, 11.0, 12.0],
        "q10": -1.0,
        "q50": 0.0,
        "q90": 1.0,
    })

    calibration = learn_quantile_calibration(
        frame=frame,
        target_column="target",
        prediction_columns=["q10", "q50", "q90"],
        smoothing_window=1,
        calibrate_median=False,
    )

    assert calibration.loc[0, "median_bias"] == 0.0
