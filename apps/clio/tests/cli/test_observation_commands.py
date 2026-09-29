from unittest.mock import AsyncMock, Mock
from types import SimpleNamespace
from clio import cli
from clio.db import session as database

import pandas as pd
import pytest
from common.schemas.observation import Observation
from clio.commands import normalize as ingestion


def session_factory(monkeypatch, module):
    session = AsyncMock()
    context = AsyncMock()
    context.__aenter__.return_value = session
    monkeypatch.setattr(module, 'get_session_factory', lambda: lambda: context)
    dispose = AsyncMock()
    monkeypatch.setattr(database, 'dispose_engine', dispose)
    return session, dispose


def test_refresh_command_only_normalizes_stored_observations(monkeypatch):
    session, dispose = session_factory(monkeypatch, ingestion)
    refresh = AsyncMock(return_value=Observation(points=[]))
    monkeypatch.setattr(ingestion, 'refresh_normalized_observations', refresh)
    cli.invoke('normalize', SimpleNamespace())
    refresh.assert_awaited_once()
    dispose.assert_awaited_once()


@pytest.mark.parametrize('selected', [None, ['kp', 'dst']])
def test_backfill_command_uses_config_for_defaults_and_explicit_selection(monkeypatch, selected):
    from datetime import UTC, datetime
    from clio.commands import backfill_observations as command
    session, dispose = session_factory(monkeypatch, command)
    config = SimpleNamespace(observations={'kp': object(), 'dst': object(), 'f10_7': object()})
    monkeypatch.setattr(command, 'load_observation_config', lambda: config)
    expected = {'failed_metrics': ['dst'], 'status': 'partial'}
    backfill = AsyncMock(return_value=expected)
    monkeypatch.setattr('clio.observations.backfill.backfill_selected', backfill)
    now = datetime(2026, 9, 28, 12, tzinfo=UTC)
    result = cli.invoke('backfill', SimpleNamespace(metrics=selected, now=now, start=None, end=None, scheduled=True))
    assert result == expected
    backfill.assert_awaited_once_with(session, config, selected or list(config.observations),
                                     now=now, start=None, end=None, raise_on_failure=False)
    dispose.assert_awaited_once()


def test_live_command_returns_per_metric_failures_to_scheduler(monkeypatch):
    from datetime import UTC, datetime
    from clio.commands import collect as command
    session, dispose = session_factory(monkeypatch, command)
    config = SimpleNamespace(observations={'kp': object(), 'dst': object()})
    monkeypatch.setattr(command, 'load_observation_config', lambda: config)
    result = {'failed_metrics': ['kp'], 'downloaded_measurements': 0}
    collect = AsyncMock(return_value=result)
    monkeypatch.setattr('clio.observations.live.collect_live', collect)
    now = datetime(2026, 9, 28, 12, tzinfo=UTC)
    args = SimpleNamespace(metrics=['kp'], now=now, scheduled=True)
    assert cli.invoke('collect', args) == result
    collect.assert_awaited_once_with(session, config, ['kp'], now=now)
    dispose.assert_awaited_once()
    args.scheduled = False
    with pytest.raises(RuntimeError, match='No live observations'):
        cli.invoke('collect', args)


@pytest.mark.parametrize('name', ['collect', 'backfill'])
def test_file_only_commands_never_open_numeric_storage(monkeypatch, name):
    from datetime import UTC, datetime
    from clio.commands import collect as fetch_live, backfill_observations
    from clio.domains.aia import collection
    command = fetch_live if name == 'collect' else backfill_observations
    numeric = Mock(side_effect=AssertionError('Opened numeric session for files'))
    monkeypatch.setattr(command, 'get_session_factory', numeric)
    monkeypatch.setattr(database, 'dispose_engine', AsyncMock())
    files = {'aia193': {'received': 0, 'restored': 0, 'retained': 1}}
    collect = AsyncMock(return_value={'status': 'complete', 'failed_metrics': [], 'files': files})
    monkeypatch.setattr(collection, 'collect_file_observations', collect)
    now = datetime(2026, 9, 28, 12, tzinfo=UTC)
    result = cli.invoke(name, SimpleNamespace(metrics=['aia193'], now=now, start=None, end=None, scheduled=False))
    assert result['files'] == files and result['downloaded_measurements'] == 0
    assert collect.call_args.args[1] == ['aia193']
    assert collect.call_args.kwargs['mode'] == ('live' if name == 'collect' else 'backfill')
    numeric.assert_not_called()


def test_mixed_backfill_commits_numeric_success_and_reports_file_failure(monkeypatch):
    from datetime import UTC, datetime
    from clio.commands import backfill_observations as command
    from clio.domains.aia import collection
    session_factory(monkeypatch, command)
    numeric = AsyncMock(return_value={'status': 'complete', 'failed_metrics': [], 'downloaded_measurements': 1})
    monkeypatch.setattr('clio.observations.backfill.backfill_selected', numeric)
    monkeypatch.setattr(collection, 'collect_file_observations', AsyncMock(return_value={
        'status': 'partial', 'failed_metrics': ['aia193'],
        'files': {'aia193': {'received': 0, 'restored': 0, 'retained': 0, 'failed': 1}}}))
    result = cli.invoke('backfill', SimpleNamespace(metrics=['v', 'aia193'], now=datetime(2026, 9, 28, tzinfo=UTC),
                                                   start=None, end=None, scheduled=True))
    assert numeric.call_args.args[2] == ['v']
    assert result['failed_metrics'] == ['aia193'] and result['downloaded_measurements'] == 1
