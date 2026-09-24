from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from common.schemas.observation import Observation
from argus_clio.commands import refresh_observations as ingestion


def session_factory(monkeypatch, module):
    session = AsyncMock()
    context = AsyncMock()
    context.__aenter__.return_value = session
    monkeypatch.setattr(module, 'get_session_factory', lambda: lambda: context)
    dispose = AsyncMock()
    monkeypatch.setattr(module, 'dispose_engine', dispose)
    return session, dispose


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
    monkeypatch.setattr(ingestion, 'upsert_measurements', upsert)
    monkeypatch.setattr(ingestion, 'load_measurements', AsyncMock(return_value=history))
    ingestion.main([])
    refresh.assert_awaited_once()
    fetch.assert_called_once()
    if missing_history:
        upsert.assert_not_awaited()
    else:
        upsert.assert_awaited_once()
        assert upsert.call_args.args[0] is session
        session.commit.assert_awaited_once()
    dispose.assert_awaited_once()
