from unittest.mock import AsyncMock, Mock
from types import SimpleNamespace
import asyncio
from clio.ingestion import collect as collection_operation
from clio.config import load_observation_config

from datetime import UTC, datetime
import pytest
from clio.ingestion import collect_numeric as live
from clio.ingestion import backfill as backfill_operation
from clio.ingestion import cleanup as cleanup_operation


def session_factory(monkeypatch, module):
    session = AsyncMock()
    context = AsyncMock()
    context.__aenter__.return_value = session
    monkeypatch.setattr(module, 'get_db_session', lambda: context)
    return session


def test_backfill_operation_uses_all_configured_observations(monkeypatch):
    session = session_factory(monkeypatch, backfill_operation)
    config = SimpleNamespace(observations={m: load_observation_config().observations[m] for m in ('kp', 'dst', 'f10_7')})
    expected = {'failed_metrics': ['dst'], 'status': 'partial'}
    backfill = AsyncMock(return_value=expected)
    monkeypatch.setattr('clio.ingestion.backfill_numeric.backfill_numeric', backfill)
    now = datetime(2026, 9, 28, 12, tzinfo=UTC)
    result = asyncio.run(backfill_operation.backfill(config, now=now))
    assert result == expected
    backfill.assert_awaited_once_with(session, config, list(config.observations),
                                     now=now, raise_on_failure=False)


def test_live_operation_returns_per_metric_failures(monkeypatch):
    session = session_factory(monkeypatch, live)
    config = SimpleNamespace(observations={m: load_observation_config().observations[m] for m in ('kp', 'dst')})
    result = {'failed_metrics': ['kp'], 'downloaded_measurements': 0}
    collect = AsyncMock(return_value=result)
    monkeypatch.setattr('clio.ingestion.collect_numeric.collect_numeric', collect)
    now = datetime(2026, 9, 28, 12, tzinfo=UTC)
    assert asyncio.run(collection_operation.collect(config, ['kp'], now=now)) == result
    collect.assert_awaited_once_with(config, ['kp'], now=now)


@pytest.mark.parametrize('name', ['collect', 'backfill'])
def test_file_only_operations_never_open_numeric_storage(monkeypatch, name):
    from clio.ingestion import collect_files, backfill_files
    collection = collect_files if name == 'collect' else backfill_files
    numeric = Mock(side_effect=AssertionError('Opened numeric session for files'))
    monkeypatch.setattr(live if name == 'collect' else backfill_operation, 'get_db_session', numeric)
    from clio.config import ClioObservations, load_observation_config
    policy = load_observation_config().observations['gong']
    files = {'gong': {'received': 0, 'restored': 0, 'retained': 1}}
    collect = AsyncMock(return_value={'status': 'complete', 'failed_metrics': [], 'files': files})
    monkeypatch.setattr(collection, 'collect_files' if name == 'collect' else 'backfill_files', collect)
    now = datetime(2026, 9, 28, 12, tzinfo=UTC)
    config = ClioObservations(observations={'gong': policy})
    result = asyncio.run(collection_operation.collect(config, ['gong'], now=now) if name == 'collect'
                         else backfill_operation.backfill(config, now=now))
    assert result['files'] == files and result['downloaded_measurements'] == 0
    assert collect.call_args.args[1] == ['gong']
    assert collect.call_args.kwargs == {'now': now}
    numeric.assert_not_called()


def test_mixed_backfill_commits_numeric_success_and_reports_file_failure(monkeypatch):
    from clio.ingestion import backfill_files as collection
    session_factory(monkeypatch, backfill_operation)
    from clio.config import ClioObservations, load_observation_config
    policies = load_observation_config().observations
    config = ClioObservations(observations={m: policies[m] for m in ('v', 'gong')})
    numeric = AsyncMock(return_value={'status': 'complete', 'failed_metrics': [], 'downloaded_measurements': 1})
    monkeypatch.setattr('clio.ingestion.backfill_numeric.backfill_numeric', numeric)
    monkeypatch.setattr(collection, 'backfill_files', AsyncMock(return_value={
        'status': 'partial', 'failed_metrics': ['gong'],
        'files': {'gong': {'received': 0, 'restored': 0, 'retained': 0, 'failed': 1}}}))
    result = asyncio.run(backfill_operation.backfill(config, now=datetime(2026, 9, 28, tzinfo=UTC)))
    assert numeric.call_args.args[2] == ['v']
    assert result['failed_metrics'] == ['gong'] and result['downloaded_measurements'] == 1


@pytest.mark.parametrize('name', ['collect', 'backfill'])
@pytest.mark.parametrize('selected', [None, ['aia_171_nrt']])
def test_common_operations_dispatch_configured_sdo_observations(monkeypatch, name, selected):
    from clio.ingestion.utils import file_collection
    from clio.ingestion.extras import donki, enlil
    from clio.config import ClioObservations, load_observation_config
    config = ClioObservations(observations={'aia_171_nrt': load_observation_config().observations['aia_171_nrt']},
                              solar_wind_events={'enabled': True})
    monkeypatch.setattr(live if name == 'collect' else backfill_operation, 'get_db_session', Mock(side_effect=AssertionError('numeric storage')))
    donki_collect, enlil_collect = AsyncMock(), AsyncMock()
    monkeypatch.setattr(donki, 'collect', donki_collect)
    monkeypatch.setattr(enlil, 'collect', enlil_collect)
    dispatch = AsyncMock(return_value={'status': 'complete', 'saved': 1})
    monkeypatch.setattr(file_collection, 'collect_observation', dispatch)
    now = datetime(2026, 10, 4, tzinfo=UTC)
    result = asyncio.run(collection_operation.collect(config, selected, now=now) if name == 'collect'
                         else backfill_operation.backfill(config, now=now))
    assert result['files']['aia_171_nrt']['saved'] == 1
    dispatch.assert_awaited_once_with(config, 'aia_171_nrt', mode='live' if name == 'collect' else 'backfill', now=now)
    assert donki_collect.await_count == enlil_collect.await_count == int(name == 'collect' and selected is None)


@pytest.mark.parametrize('selected', [None, ['donki'], ['enlil'], ['donki', 'enlil', 'donki']])
@pytest.mark.parametrize('failed', [False, True])
@pytest.mark.parametrize('enabled', [True, False])
def test_catalog_collection_selection_and_independent_failures(monkeypatch, selected, failed, enabled):
    from clio.ingestion.extras import donki, enlil
    config = SimpleNamespace(observations={}, sdo_images=SimpleNamespace(enabled=False),
                             solar_wind_events=SimpleNamespace(enabled=enabled, lookback_days=30))
    monkeypatch.setattr(live, 'get_db_session', Mock(side_effect=AssertionError('numeric storage')))
    donki_collect = AsyncMock(side_effect=RuntimeError('DONKI failed') if failed else None)
    enlil_collect = AsyncMock()
    monkeypatch.setattr(donki, 'collect', donki_collect)
    monkeypatch.setattr(enlil, 'collect', enlil_collect)
    now = datetime(2026, 10, 27, tzinfo=UTC)
    result = asyncio.run(collection_operation.collect(config, selected, now=now))
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


def test_cleanup_expires_numeric_data_and_runs_file_retention(storage, monkeypatch):
    from clio.ingestion.extras import sdo
    from clio.config import ClioObservations
    session = session_factory(monkeypatch, cleanup_operation)
    session.execute.return_value = Mock(rowcount=3)
    config = ClioObservations(observations={},
                              retention={'live_days': 8, 'normalized_days': 90})
    now = datetime(2026, 10, 27, tzinfo=UTC)
    result = asyncio.run(cleanup_operation.cleanup(config, now))
    assert result['live_removed'] == result['normalized_removed'] == 3
    from clio.providers.donki import ENDPOINTS
    assert set(result['files']) == {'enlil', *('donki.'+endpoint for endpoint in ENDPOINTS)}
    from datetime import timedelta
    live_delete, hourly_delete = [call.args[0].compile() for call in session.execute.call_args_list]
    assert live_delete.params['observed_at_1'] == now - timedelta(days=8)
    assert hourly_delete.params['observed_at_1'] == now - timedelta(days=90)
    assert 'omni.hourly' not in live_delete.params['source_product_1']
    assert 'swpc.rtsw_mag' in live_delete.params['source_product_1']
    session.commit.assert_awaited_once()


@pytest.mark.parametrize('name,mode', [('collect', 'live'), ('backfill', 'backfill')])
def test_file_entrypoints_use_their_own_mode_and_preserve_failures(monkeypatch, name, mode):
    from clio.ingestion import collect_files, backfill_files
    from clio.ingestion.extras import gong
    from clio.config import ClioObservations, load_observation_config
    config = ClioObservations(observations={'gong': load_observation_config().observations['gong']})
    report = {'status': 'partial', 'failed_metrics': ['gong'], 'files': {'gong': {'failed': 1}}}
    collect = AsyncMock(return_value=report)
    monkeypatch.setattr(gong, 'collect', collect)
    operation = collect_files.collect_files if name == 'collect' else backfill_files.backfill_files
    now = datetime(2026, 10, 4, tzinfo=UTC)
    result = asyncio.run(operation(config, ['gong'], now=now))
    assert result['status'] == report['status'] and result['failed_metrics'] == ['gong']
    assert result['files']['gong']['failed'] == 1
    collect.assert_awaited_once_with(config, mode=mode, now=now)
