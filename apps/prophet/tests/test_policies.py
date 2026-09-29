from datetime import UTC, datetime, timedelta
import pytest

from argus_prophet.config import InputPolicy, ProphetConfig
from common.schemas.forecast_inputs import ForecastInputs, GONGFeatureFrame
from common.schemas.observation import Observation


def test_gong_policy_rejects_missing_stale_and_future_inputs():
    now = datetime(2026, 9, 29, tzinfo=UTC)
    inputs = ForecastInputs(as_of=now, read_at=now, observations=Observation(points=[]))
    policy = InputPolicy(max_gong_age_hours=3)
    with pytest.raises(ValueError, match='GONG'):
        policy.validate_inputs(inputs)
    inputs.gong = GONGFeatureFrame(observed_at=now-timedelta(hours=3), available_at=now,
                                   sha256='a'*64, source_product='gong.live', features={'field': 1.})
    policy.validate_inputs(inputs)
    inputs.gong.observed_at -= timedelta(seconds=1)
    with pytest.raises(ValueError, match='GONG'):
        policy.validate_inputs(inputs)
    inputs.gong.observed_at = now
    inputs.gong.available_at += timedelta(seconds=1)
    with pytest.raises(ValueError, match='GONG'):
        policy.validate_inputs(inputs)


@pytest.mark.parametrize('settings', [
    {'calculation_timeout_seconds': 0}, {'shutdown_grace_seconds': 600},
    {'inputs': {'unknown': {}}}, {'inputs': {'dst': {'max_normalized_age_hours': float('nan')}}},
    {'verification': {'days': 26}},
])
def test_invalid_operational_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        ProphetConfig.model_validate(settings)


@pytest.mark.parametrize('settings', [
    {'schedules': {'unknown': {}}}, {'schedules': {'dst': {'every_minutes': 0}}},
    {'schedules': {'dst': {'every_minutes': 5, 'offset_minutes': 5}}},
    {'schedules': {'dst': {'offset_minutes': -1}}}, {'max_parallel_products': 0},
])
def test_invalid_product_schedules_are_rejected(settings):
    with pytest.raises(ValueError):
        ProphetConfig.model_validate(settings)
