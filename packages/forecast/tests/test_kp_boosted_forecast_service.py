from datetime import UTC, datetime

import numpy as np
from common.adapters import observations_to_dataframe
from common.schemas.observation import Observation, ObservationPoint
from forecast.inference.KpBoostedForecastService import KpBoostedForecastService


class ConstantProbabilityModel:
    classes_ = np.array([0, 1])

    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, features):
        positive = np.full(len(features), self.probability)
        return np.column_stack([1 - positive, positive])


def _model_bundle() -> dict:
    cuts = tuple(range(1, 7))
    buckets = {"0_3h": (0, 3), "3_6h": (3, 6)}
    probabilities = {cut: 1 - cut / 10 for cut in cuts}
    return {
        "models": {
            bucket: {
                cut: ConstantProbabilityModel(probabilities[cut]) for cut in cuts
            }
            for bucket in buckets
        },
        "calibrators": {
            bucket: {
                cut: {
                    "method": "constant",
                    "estimator": None,
                    "constant": probabilities[cut],
                }
                for cut in cuts
            }
            for bucket in buckets
        },
        "ordinal_cuts": cuts,
        "horizon_buckets": buckets,
        "hht_features": [],
        "feature_columns_by_bucket": {
            bucket: ["kp", "lead_hours"] for bucket in buckets
        },
    }


def _observations() -> Observation:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return Observation(
        points=[
            ObservationPoint(
                issue_time=start.replace(hour=hour),
                bx=1.0,
                by=2.0,
                bz=-3.0,
                v=400.0,
                n=5.0,
                t=100_000.0,
                kp=2,
                dst=-10,
                ap=7,
                f10_7=120,
            )
            for hour in range(12)
        ]
    )


def test_forecast_uses_bucketed_ordinal_models_and_calibrators() -> None:
    service = KpBoostedForecastService({"boosted": _model_bundle()})

    forecast = service.forecast(_observations())

    assert len(forecast.points) == 6
    assert [point.lead_hours for point in forecast.points] == list(range(1, 7))
    for point in forecast.points:
        assert point.p_kp_4 == 0.6
        assert point.p_kp_5 == 0.5
        assert point.p_kp_6 == 0.4
        assert point.p_kp_7 == 0.0


def test_service_accepts_notebook_bundle_directly() -> None:
    service = KpBoostedForecastService(_model_bundle())

    assert service.model_bundle["ordinal_cuts"] == tuple(range(1, 7))


def test_hht_summary_is_computed_from_causal_window() -> None:
    values = np.sin(np.arange(168) / 5) + np.sin(np.arange(168) / 17)

    features = KpBoostedForecastService._hht_window_summary(values, "bz")

    assert np.isfinite(features["hht_bz_imf1_energy"])
    assert np.isfinite(features["hht_bz_residue"])


def test_physical_features_match_training_units() -> None:
    history = KpBoostedForecastService._add_issue_time_features(
        observations_to_dataframe(_observations())
    )

    assert history.iloc[-1]["bt"] == np.sqrt(13)
    assert history.iloc[-1]["dynamic_pressure"] == 800_000
