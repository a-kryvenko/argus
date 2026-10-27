from unittest.mock import AsyncMock, Mock
from types import SimpleNamespace
from clio import cli
from common.db import session as database

from datetime import UTC, datetime
import pytest
from common.schemas.observation import Observation
from clio.commands import normalize as ingestion
from clio.ingestion import numeric_live as live
from clio.ingestion import backfill as backfill_operation


def session_factory(monkeypatch, module):
    session = AsyncMock()
    context = AsyncMock()
    context.__aenter__.return_value = session
    target = live if module.__name__.endswith('.collect') else backfill_operation if module.__name__.endswith('.backfill') else module
    monkeypatch.setattr(target, 'get_db_session', lambda: context)
    dispose = AsyncMock()
    monkeypatch.setattr(database, 'dispose_engine', dispose)
    return session, dispose


def test_refresh_command_only_normalizes_stored_observations(monkeypatch):
    session, dispose = session_factory(monkeypatch, ingestion)
    refresh = AsyncMock(return_value=Observation(points=[]))
    monkeypatch.setattr(ingestion, 'refresh_normalized_observations', refresh)
    cli.invoke('normalize', SimpleNamespace(now=datetime.now(UTC)))
    refresh.assert_awaited_once()
    dispose.assert_awaited_once()


def test_backfill_command_uses_all_configured_observations(monkeypatch):
    from datetime import UTC, datetime
    from clio.commands import backfill as command
    session, dispose = session_factory(monkeypatch, command)
    config = SimpleNamespace(sdo_images=SimpleNamespace(enabled=False), observations={'kp': object(), 'dst': object(), 'f10_7': object()})
    monkeypatch.setattr(command, 'load_observation_config', lambda: config)
    expected = {'failed_metrics': ['dst'], 'status': 'partial'}
    backfill = AsyncMock(return_value=expected)
    monkeypatch.setattr('clio.ingestion.numeric_backfill.backfill_numeric', backfill)
    now = datetime(2026, 9, 28, 12, tzinfo=UTC)
    result = cli.invoke('backfill', SimpleNamespace(now=now))
    assert result == expected
    backfill.assert_awaited_once_with(session, config, list(config.observations),
                                     now=now, raise_on_failure=False)
    dispose.assert_awaited_once()


def test_live_command_returns_per_metric_failures(monkeypatch):
    from datetime import UTC, datetime
    from clio.commands import collect as command
    session, dispose = session_factory(monkeypatch, command)
    config = SimpleNamespace(sdo_images=SimpleNamespace(enabled=False), observations={'kp': object(), 'dst': object()})
    monkeypatch.setattr(command, 'load_observation_config', lambda: config)
    result = {'failed_metrics': ['kp'], 'downloaded_measurements': 0}
    collect = AsyncMock(return_value=result)
    monkeypatch.setattr('clio.ingestion.numeric_live.collect_live', collect)
    now = datetime(2026, 9, 28, 12, tzinfo=UTC)
    args = SimpleNamespace(metrics=['kp'], now=now, batch=True)
    assert cli.invoke('collect', args) == result
    collect.assert_awaited_once_with(config, ['kp'], now=now)
    dispose.assert_awaited_once()



@pytest.mark.parametrize('name', ['collect', 'backfill'])
def test_file_only_commands_never_open_numeric_storage(monkeypatch, name):
    from datetime import UTC, datetime
    from clio.commands import collect as fetch_live, backfill
    from clio.ingestion import files as collection
    command = fetch_live if name == 'collect' else backfill
    numeric = Mock(side_effect=AssertionError('Opened numeric session for files'))
    monkeypatch.setattr(live if name == 'collect' else backfill_operation, 'get_db_session', numeric)
    monkeypatch.setattr(database, 'dispose_engine', AsyncMock())
    from clio.config import ClioObservations, load_observation_config
    policy = load_observation_config().observations['gong']
    monkeypatch.setattr(command, 'load_observation_config', lambda: ClioObservations(observations={'gong': policy}))
    files = {'gong': {'received': 0, 'restored': 0, 'retained': 1}}
    collect = AsyncMock(return_value={'status': 'complete', 'failed_metrics': [], 'files': files})
    monkeypatch.setattr(collection, 'collect_file_observations', collect)
    now = datetime(2026, 9, 28, 12, tzinfo=UTC)
    result = cli.invoke(name, SimpleNamespace(metrics=['gong'], now=now) if name == 'collect' else SimpleNamespace(now=now))
    assert result['files'] == files and result['downloaded_measurements'] == 0
    assert collect.call_args.args[1] == ['gong']
    assert collect.call_args.kwargs['mode'] == ('live' if name == 'collect' else 'backfill')
    numeric.assert_not_called()


def test_mixed_backfill_commits_numeric_success_and_reports_file_failure(monkeypatch):
    from datetime import UTC, datetime
    from clio.commands import backfill as command
    from clio.ingestion import files as collection
    session_factory(monkeypatch, command)
    from clio.config import ClioObservations, load_observation_config
    policies = load_observation_config().observations
    config = ClioObservations(observations={m: policies[m] for m in ('v', 'gong')})
    monkeypatch.setattr(command, 'load_observation_config', lambda: config)
    numeric = AsyncMock(return_value={'status': 'complete', 'failed_metrics': [], 'downloaded_measurements': 1})
    monkeypatch.setattr('clio.ingestion.numeric_backfill.backfill_numeric', numeric)
    monkeypatch.setattr(collection, 'collect_file_observations', AsyncMock(return_value={
        'status': 'partial', 'failed_metrics': ['gong'],
        'files': {'gong': {'received': 0, 'restored': 0, 'retained': 0, 'failed': 1}}}))
    result = cli.invoke('backfill', SimpleNamespace(now=datetime(2026, 9, 28, tzinfo=UTC)))
    assert numeric.call_args.args[2] == ['v']
    assert result['failed_metrics'] == ['gong'] and result['downloaded_measurements'] == 1


@pytest.mark.parametrize('name,mode', [('collect', 'live'), ('backfill', 'warmup')])
@pytest.mark.parametrize('selected', [None, ['sdo']])
def test_common_commands_include_sdo_without_numeric_storage(monkeypatch, name, mode, selected):
    from datetime import UTC, datetime
    from clio.commands import collect, backfill
    from clio.ingestion import sdo_images
    from clio.ingestion import donki, enlil
    from clio.config import ClioObservations
    command = collect if name == 'collect' else backfill
    config = ClioObservations(observations={}, sdo_images={'enabled': True}, solar_wind_events={'enabled': True})
    monkeypatch.setattr(command, 'load_observation_config', lambda: config)
    monkeypatch.setattr(live if name == 'collect' else backfill_operation, 'get_db_session', Mock(side_effect=AssertionError('numeric storage')))
    monkeypatch.setattr(database, 'dispose_engine', AsyncMock())
    donki_collect, enlil_collect = AsyncMock(), AsyncMock()
    monkeypatch.setattr(donki, 'collect', donki_collect)
    monkeypatch.setattr(enlil, 'collect', enlil_collect)
    cycle = Mock(return_value={'saved': 1, 'failed': 0})
    monkeypatch.setattr(sdo_images, 'cycle', cycle)
    now = datetime(2026, 10, 4, tzinfo=UTC)
    result = cli.invoke(name, SimpleNamespace(metrics=selected, now=now) if name == 'collect' else SimpleNamespace(now=now))
    assert result['sdo']['saved'] == 1
    assert cycle.call_args.args[1] == mode
    assert cycle.call_args.kwargs['full'] is True
    assert cycle.call_args.kwargs['clock']() == now
    assert donki_collect.await_count == enlil_collect.await_count == int(name == 'collect' and selected is None)


@pytest.mark.parametrize('selected', [None, ['donki'], ['enlil'], ['donki', 'enlil', 'donki']])
@pytest.mark.parametrize('failed', [False, True])
@pytest.mark.parametrize('enabled', [True, False])
def test_catalog_collection_selection_and_independent_failures(monkeypatch, selected, failed, enabled):
    from datetime import UTC, datetime
    from clio.commands import collect
    from clio.ingestion import donki, enlil
    config = SimpleNamespace(observations={}, sdo_images=SimpleNamespace(enabled=False),
                             solar_wind_events=SimpleNamespace(enabled=enabled, lookback_days=30))
    monkeypatch.setattr(collect, 'load_observation_config', lambda: config)
    monkeypatch.setattr(live, 'get_db_session', Mock(side_effect=AssertionError('numeric storage')))
    monkeypatch.setattr(database, 'dispose_engine', AsyncMock())
    donki_collect = AsyncMock(side_effect=RuntimeError('DONKI failed') if failed else None)
    enlil_collect = AsyncMock()
    monkeypatch.setattr(donki, 'collect', donki_collect)
    monkeypatch.setattr(enlil, 'collect', enlil_collect)
    now = datetime(2026, 10, 27, tzinfo=UTC)
    result = cli.invoke('collect', SimpleNamespace(metrics=selected, now=now))
    chosen = {'donki', 'enlil'} if selected is None else set(selected)
    if enabled and 'donki' in chosen:
        donki_collect.assert_awaited_once_with(config.solar_wind_events, now=now)
    else:
        donki_collect.assert_not_awaited()
    if enabled and 'enlil' in chosen:
        enlil_collect.assert_awaited_once_with(now=now)
        assert result['enlil']['status'] == 'complete'
    else:
        enlil_collect.assert_not_awaited()
    if not enabled:
        assert all(result[kind] == {'status': 'disabled'} for kind in chosen)
    assert result['failed_metrics'] == (['donki'] if enabled and failed and 'donki' in chosen else [])
    assert result['status'] == ('partial' if result['failed_metrics'] else 'complete')


def test_cleanup_removes_expired_data_and_runs_sdo_retention(monkeypatch):
    from clio.commands import cleanup
    from clio.ingestion import sdo_images
    from clio.config import ClioObservations
    session, dispose = session_factory(monkeypatch, cleanup)
    session.execute.return_value = Mock(rowcount=3)
    config = ClioObservations(observations={}, sdo_images={'enabled': True},
                              retention={'live_days': 8, 'normalized_days': 90})
    monkeypatch.setattr(cleanup, 'load_observation_config', lambda: config)
    cycle = Mock(return_value={'removed': 4})
    monkeypatch.setattr(sdo_images, 'cycle', cycle)
    now = datetime(2026, 10, 27, tzinfo=UTC)
    result = cli.invoke('cleanup', SimpleNamespace(now=now))
    assert result == {'live_removed': 3, 'normalized_removed': 3, 'sdo': {'removed': 4}}
    from datetime import timedelta
    live_delete, hourly_delete = [call.args[0].compile() for call in session.execute.call_args_list]
    assert live_delete.params['observed_at_1'] == now - timedelta(days=8)
    assert hourly_delete.params['observed_at_1'] == now - timedelta(days=90)
    assert 'omni.hourly' not in live_delete.params['source_product_1']
    assert 'swpc.rtsw_mag' in live_delete.params['source_product_1']
    assert cycle.call_args.args[1] == 'cleanup'
    assert cycle.call_args.kwargs['clock']() == now
    session.commit.assert_awaited_once()
    dispose.assert_awaited_once()


@pytest.mark.parametrize('name', ['collect', 'backfill'])
def test_commands_delegate_to_one_ingestion_operation(monkeypatch, name):
    from importlib import import_module
    command = import_module(f'clio.commands.{name}')
    operation = import_module(f'clio.ingestion.{name}')
    config = object()
    report = {'status': 'complete', 'failed_metrics': []}
    run = AsyncMock(return_value=report)
    monkeypatch.setattr(command, 'load_observation_config', lambda: config)
    monkeypatch.setattr(operation, name, run)
    monkeypatch.setattr(database, 'dispose_engine', AsyncMock())
    now = datetime(2026, 10, 27, tzinfo=UTC)
    args = SimpleNamespace(now=now, metrics=['gong']) if name == 'collect' else SimpleNamespace(now=now)
    assert cli.invoke(name, args) == report
    if name == 'collect':
        run.assert_awaited_once_with(config, ['gong'], now=now)
    else:
        run.assert_awaited_once_with(config, now=now)
