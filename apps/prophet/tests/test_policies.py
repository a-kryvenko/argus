from datetime import UTC, datetime
import pytest

from prophet.config import InputPolicy, ProphetConfig
from forecast.inputs import ForecastInputs
from common.schemas.observation import Observation


def test_default_policy_accepts_inputs_without_retired_gong_features():
    now = datetime(2026, 9, 29, tzinfo=UTC)
    inputs = ForecastInputs(as_of=now, read_at=now, observations=Observation(points=[]))
    policy = InputPolicy()
    policy.validate_inputs(inputs)
    assert 'max_gong_age_hours' not in policy.model_dump()


@pytest.mark.parametrize('settings', [
    {'calculation_timeout_seconds': 0}, {'shutdown_grace_seconds': 600},
    {'inputs': {'unknown': {}}}, {'inputs': {'dst': {'max_normalized_age_hours': float('nan')}}},
    {'verification_days': 0},
    {'verification_timeout_seconds': 0}, {'verification_timeout_seconds': 541},
    {'inputs': {'hmf': {'max_gong_age_hours': 3}}},
])
def test_invalid_operational_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        ProphetConfig.model_validate(settings)


@pytest.mark.parametrize('settings', [
    {'schedules': {'unknown': {}}}, {'schedules': {'dst': {'every_minutes': 0}}},
    {'schedules': {'dst': {'every_minutes': 5, 'offset_minutes': 5}}},
    {'schedules': {'dst': {'offset_minutes': -1}}}, {'max_parallel_products': 0},
])
def test_retired_scheduler_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        ProphetConfig.model_validate(settings)
