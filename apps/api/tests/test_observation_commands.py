import asyncio
import importlib
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from common.schemas.observation import Observation
from app.commands import _sensor_observations as reader
from app.commands import refresh_observations as ingestion
from app.services import sensor_observations, density_observations
from app.services.forecast_products import ArtifactNotReadyError


def session_factory(monkeypatch, module):
    session = AsyncMock()
    context = AsyncMock()
    context.__aenter__.return_value = session
    monkeypatch.setattr(module, 'get_session_factory', lambda: lambda: context)
    dispose = AsyncMock()
    monkeypatch.setattr(module, 'dispose_engine', dispose)
    return session, dispose


def test_forecast_reader_does_not_ingest_on_empty_database(monkeypatch):
    session, dispose = session_factory(monkeypatch, reader)
    loader = AsyncMock(return_value=Observation(points=[]))
    refresh = AsyncMock(side_effect=AssertionError('Forecast must not ingest'))
    monkeypatch.setattr(reader, 'load_normalized_observations', loader)
    monkeypatch.setattr(sensor_observations, 'refresh_normalized_observations', refresh)
    with pytest.raises(RuntimeError, match='refresh_observations first'):
        reader.load_sensor_observations()
    loader.assert_awaited_once()
    refresh.assert_not_awaited()
    session.commit.assert_not_awaited()
    dispose.assert_awaited_once()


@pytest.mark.parametrize('name', [
    'generate_forecast', 'generate_wind_forecast',
    'generate_hmf_forecast', 'generate_kp_forecast',
])
def test_forecasts_pass_stored_observations_to_director(monkeypatch, name):
    command = importlib.import_module(f'app.commands.{name}')
    stored = object()
    monkeypatch.setattr(command, 'load_sensor_observations', Mock(return_value=stored))
    director = Mock()
    monkeypatch.setattr(command, 'ForecastDirector', Mock(return_value=director))
    monkeypatch.setattr(command.ForecastServiceRegistry, 'get', Mock())
    if name == 'generate_forecast':
        monkeypatch.setattr(command, 'generate_density', Mock())
    command.main()
    assert director.refresh_forecasts.call_args.args[1] is stored


@pytest.mark.parametrize('missing_history', [False, True])
def test_ingestion_refreshes_live_and_persists_optional_history(monkeypatch, missing_history):
    session, dispose = session_factory(monkeypatch, ingestion)
    refresh = AsyncMock(return_value=Observation(points=[]))
    history = pd.DataFrame(columns=['metric', 'value', 'observed_at'])
    fetch = Mock(side_effect=FileNotFoundError('calibration') if missing_history else None,
                 return_value=history)
    upsert = AsyncMock()
    monkeypatch.setattr(ingestion, 'refresh_normalized_observations', refresh)
    monkeypatch.setattr(ingestion, 'load_density_history', fetch)
    monkeypatch.setattr(ingestion, '_upsert_measurements', upsert)
    monkeypatch.setattr(ingestion, '_load_measurements', AsyncMock(return_value=history))
    ingestion.main()
    refresh.assert_awaited_once()
    fetch.assert_called_once()
    if missing_history:
        upsert.assert_not_awaited()
    else:
        upsert.assert_awaited_once()
        assert upsert.call_args.args[0] is session
        session.commit.assert_awaited_once()
    dispose.assert_awaited_once()


def test_density_missing_inputs_do_not_download_history(monkeypatch):
    from forecast_core import api
    fetch = Mock(side_effect=AssertionError('Forecast must not download history'))
    monkeypatch.setattr(api, 'load_density_history', fetch)
    session = AsyncMock()
    session.execute.return_value = Mock(all=Mock(return_value=[]))
    monkeypatch.setattr(density_observations, 'observed_driver_frame',
                        Mock(side_effect=ArtifactNotReadyError('s10 history missing')))
    from datetime import UTC, datetime
    with pytest.raises(ArtifactNotReadyError, match='s10'):
        asyncio.run(density_observations.load_density_drivers(session, datetime.now(UTC)))
    fetch.assert_not_called()


def test_reader_returns_stored_data_without_writes(monkeypatch):
    from datetime import UTC, datetime
    from common.schemas.observation import ObservationPoint

    session, dispose = session_factory(monkeypatch, reader)
    stored = Observation(points=[ObservationPoint(
        issue_time=datetime.now(UTC), bx=1, by=2, bz=-3, v=420, n=5,
        t=100000, kp=2, dst=-10, ap=5, f10_7=120,
    )])
    monkeypatch.setattr(reader, 'load_normalized_observations', AsyncMock(return_value=stored))
    assert reader.load_sensor_observations() is stored
    session.commit.assert_not_awaited()
    dispose.assert_awaited_once()
