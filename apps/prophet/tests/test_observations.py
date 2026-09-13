from datetime import UTC, datetime
import importlib
from unittest.mock import Mock

import httpx
import pytest
from common.schemas.forecast_inputs import ForecastInputs
from common.schemas.observation import Observation, ObservationPoint
from argus_prophet import observations

NOW = datetime(2026, 9, 12, 12, tzinfo=UTC)


def stored_inputs():
    point = ObservationPoint(issue_time=NOW, bx=1, by=2, bz=-3, v=420, n=5,
                             t=100000, kp=2, dst=-10, ap=5, f10_7=120)
    return ForecastInputs(as_of=NOW, read_at=NOW, observations=Observation(points=[point]))


def configure(monkeypatch, handler):
    monkeypatch.setenv('OBSERVATIONS_URL', 'http://observations')
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'test-token')
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(observations.httpx, 'Client', lambda **kwargs: client)


def test_reads_versioned_owner_contract(monkeypatch):
    def handler(request):
        assert request.url.path == '/internal/v1/observations/forecast-inputs'
        assert request.headers['Authorization'] == 'Bearer test-token'
        assert request.url.params['as_of'] == NOW.isoformat()
        return httpx.Response(200, json=stored_inputs().model_dump(mode='json'))
    configure(monkeypatch, handler)
    assert observations.load_inputs(NOW).observations.points[0].v == 420


@pytest.mark.parametrize('status', [401, 503])
def test_owner_failure_is_propagated_without_fallback(monkeypatch, status):
    configure(monkeypatch, lambda _: httpx.Response(status))
    with pytest.raises(httpx.HTTPStatusError):
        observations.load_inputs(NOW)


def test_empty_inputs_fail_without_ingestion(monkeypatch):
    payload = stored_inputs().model_dump(mode='json')
    payload['observations']['points'] = []
    configure(monkeypatch, lambda _: httpx.Response(200, json=payload))
    with pytest.raises(RuntimeError, match='check ingestion'):
        observations.load_inputs(NOW)


def test_rejects_incompatible_contract(monkeypatch):
    payload = stored_inputs().model_dump(mode='json')
    payload['schema_version'] = 2
    configure(monkeypatch, lambda _: httpx.Response(200, json=payload))
    with pytest.raises(ValueError):
        observations.load_inputs(NOW)


@pytest.mark.parametrize('name', ['generate_wind_forecast', 'generate_hmf_forecast', 'generate_kp_forecast'])
def test_product_commands_use_owner_observations(monkeypatch, name):
    command = importlib.import_module(f'argus_prophet.commands.{name}')
    stored = stored_inputs().observations
    monkeypatch.setattr(command, 'load_sensor_observations', Mock(return_value=stored))
    director = Mock()
    monkeypatch.setattr(command, 'ForecastDirector', Mock(return_value=director))
    monkeypatch.setattr(command.ForecastServiceRegistry, 'get', Mock())
    command.main()
    assert director.refresh_forecasts.call_args.args[1] is stored


def test_full_run_shares_one_read_with_density(monkeypatch):
    from argus_prophet.commands import generate_forecast as command
    inputs = stored_inputs()
    read = Mock(return_value=inputs)
    monkeypatch.setattr(command, 'load_inputs', read)
    director = Mock()
    monkeypatch.setattr(command, 'ForecastDirector', Mock(return_value=director))
    monkeypatch.setattr(command.ForecastServiceRegistry, 'get', Mock())
    density = Mock()
    monkeypatch.setattr(command, 'generate_density', density)
    command.main()
    read.assert_called_once_with()
    assert director.refresh_forecasts.call_args.args[1] is inputs.observations
    density.assert_called_once_with(inputs)
