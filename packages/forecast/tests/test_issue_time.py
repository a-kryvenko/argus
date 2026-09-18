from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from common.schemas.observation import Observation, ObservationPoint
from forecast.inference._forecast_service import QuantileForecastService


class ConstantModel:
    def predict(self, frame):
        return np.full(len(frame), 5.0)


class Service(QuantileForecastService):
    target_name = 'v'

    def _build_features(self, frame):
        return frame


def test_forecast_uses_explicit_issue_time_instead_of_last_observation():
    issue = datetime(2026, 9, 18, 12, 10, 5, 123, tzinfo=UTC)
    observations = Observation(points=[ObservationPoint(
        issue_time=issue - timedelta(hours=3), bx=1, by=2, bz=-3,
        v=400, n=5, t=100000, kp=2, ap=5, dst=-10, f10_7=120)])
    service = Service({'lead_hours': 2, 'buckets': [(2, 'short')],
                       'feature_columns': ['v', 'lead_hours'],
                       'models': {('short', q): ConstantModel() for q in ('q10', 'q50', 'q90')}})
    first = service.forecast(observations, issue_time=issue)
    assert first.issue_time == issue
    assert first.points[0].valid_time == issue.replace(hour=13, minute=0, second=0, microsecond=0)
    assert service.forecast(observations, issue_time=issue) == first
    with pytest.raises(ValueError, match='timezone-aware'):
        service.forecast(observations, issue_time=issue.replace(tzinfo=None))
