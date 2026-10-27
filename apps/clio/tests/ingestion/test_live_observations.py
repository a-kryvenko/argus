import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from sqlalchemy.dialects import postgresql

from clio.config import load_observation_config
from clio.ingestion import live as loaders
from clio.ingestion import numeric_live as live
from clio.observations.measurements import store_native

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def frame(*rows):
    return pd.DataFrame(rows, columns=['metric', 'value', 'observed_at']).assign(received_at=NOW)


def policies(**sources):
    return {metric: SimpleNamespace(sources=SimpleNamespace(live=names)) for metric, names in sources.items()}


def test_one_fetch_for_magnetic_and_plasma_with_separate_provenance():
    wind = AsyncMock(return_value=frame(('bx', -3., NOW), ('v', 450., NOW)))
    result, report = asyncio.run(live.fetch_live(['bx', 'v'], policies(bx=['mag'], v=['plasma']),
                                                {'mag': wind, 'plasma': wind}, NOW))
    wind.assert_awaited_once()
    assert result.source_product.tolist() == ['mag', 'plasma']
    assert report['failed_metrics'] == []


def test_fallback_fills_missing_slots_without_replacing_primary_values():
    primary = AsyncMock(return_value=frame(('v', 400., NOW - timedelta(hours=1)),
                                               ('n', 5., NOW)))
    fallback = AsyncMock(return_value=frame(('v', 450., NOW), ('n', 99., NOW)))
    result, report = asyncio.run(live.fetch_live(['v', 'n'], policies(v=['p', 'f'], n=['p', 'f']),
                                                {'p': primary, 'f': fallback}, NOW))
    assert result[result.metric == 'n'].value.tolist() == [5.]
    assert result[result.metric == 'v'].value.tolist() == [400., 450.]
    assert report['failed_metrics'] == []


def test_shared_failure_is_not_fetched_twice_and_other_sources_succeed():
    failed = AsyncMock(side_effect=OSError('offline'))
    good = AsyncMock(return_value=frame(('dst', -70., NOW)))
    result, report = asyncio.run(live.fetch_live(['bx', 'v', 'dst'], policies(bx=['mag'], v=['plasma'], dst=['dst']),
                                                {'mag': failed, 'plasma': failed, 'dst': good}, NOW))
    failed.assert_awaited_once()
    assert result.value.tolist() == [-70.]
    assert set(report['failed_metrics']) == {'bx', 'v'}
    assert [a['error'] for a in report['source_attempts'] if 'error' in a] == ['offline', 'offline']


def test_freshness_quality_and_future_times_are_checked_per_metric():
    source = AsyncMock(return_value=frame(
        ('v', 400., NOW - timedelta(hours=1)), ('v', 500., NOW + timedelta(minutes=1)),
        ('n', -1e31, NOW), ('kp', 10., NOW), ('dst', -80., NOW - timedelta(hours=2)),
        ('f10_7', 150., NOW - timedelta(days=1))))
    metrics = ['v', 'n', 'kp', 'dst', 'f10_7']
    result, report = asyncio.run(live.fetch_live(metrics, policies(**{m: ['source'] for m in metrics}),
                                                {'source': source}, NOW))
    assert set(report['failed_metrics']) == {'v', 'n', 'kp'}
    assert result[result.metric == 'v'].value.tolist() == [400.]
    assert result[result.metric == 'dst'].value.tolist() == [-80.]


def test_independent_feeds_start_concurrently():
    async def scenario():
        ready = asyncio.Event()
        entered = []

        async def fetch(metric, start, now):
            entered.append(metric)
            if len(entered) == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), timeout=1)
            return frame((metric, 1., NOW))

        sources = {m: lambda start, now, m=m: fetch(m, start, now) for m in ('v', 'dst')}
        _, report = await live.fetch_live(['v', 'dst'], policies(v=['v'], dst=['dst']), sources, NOW)
        assert report['failed_metrics'] == []

    asyncio.run(scenario())


def test_geomagnetic_loader_returns_native_values_without_storage(monkeypatch):
    received = NOW - timedelta(minutes=1)
    fetch = Mock(return_value=[dict(interval_start=NOW, value=3.33, quality='unverified',
                                   received_at=received, raw={'a_running': 18})])
    monkeypatch.setattr(loaders.geomagnetic_loader, 'fetch_records', fetch)
    result = loaders.load_geomagnetic('kp', NOW - timedelta(days=1), NOW)
    fetch.assert_called_once_with('kp')
    assert result.value.tolist() == [3.33, 18.]
    assert set(result.received_at) == {received}
    assert set(result.source_product) == {'swpc.kp'}


def test_native_error_propagates(monkeypatch):
    monkeypatch.setattr(loaders.geomagnetic_loader, 'fetch_records', Mock(side_effect=OSError('offline')))
    with pytest.raises(OSError):
        loaders.load_geomagnetic('dst', NOW - timedelta(days=1), NOW)


def test_live_saves_partial_results_without_normalization(monkeypatch):
    config = load_observation_config()
    collect = AsyncMock(return_value=frame(('v', 450., NOW)))
    monkeypatch.setattr(live, 'collect_product', collect)
    report = asyncio.run(live.collect_live(config, ['v', 'n'], now=NOW))
    assert report['failed_metrics'] == ['n']
    assert report['downloaded_measurements'] == 1
    collect.assert_awaited_once()
    assert collect.call_args.args[0] == 'swpc.rtsw_plasma'
    assert collect.call_args.args[1]['v'] == ['swpc.rtsw_plasma']


def test_live_noop_does_not_refresh_receipt_and_update_requires_priority():
    session = AsyncMock()
    session.execute.return_value = Mock(all=lambda: [])
    records = frame(('v', 450., NOW)).assign(source_product='backup', quality=None).to_dict('records')
    asyncio.run(store_native(session, records, source_priorities={'v': ['primary', 'backup']}))
    session.execute.assert_awaited_once()
    compiled = session.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert 'source_product IS DISTINCT FROM' in sql
    # A backup may correct its own observations, but not a known primary value.
    lists = [value for value in compiled.params.values() if isinstance(value, list)]
    assert ['backup'] in lists and ['primary', 'backup'] in lists


def test_live_receipt_uses_actual_changed_record_time():
    session = AsyncMock()
    received = NOW - timedelta(seconds=20)
    session.execute.return_value = Mock(all=lambda: [('v', NOW - timedelta(minutes=1), received)])
    records = frame(('v', 450., NOW - timedelta(minutes=1))).assign(
        source_product='primary', received_at=received, quality=None).to_dict('records')
    asyncio.run(store_native(session, records, source_priorities={'v': ['primary']}))
    assert session.execute.await_count == 1
    receipt = session.execute.call_args_list[0].args[0].compile(dialect=postgresql.dialect())
    assert receipt.params['received_at_m0'] == received


def storage(monkeypatch):
    session, marker = AsyncMock(), AsyncMock()
    session.execute.return_value = Mock(scalar_one=Mock(return_value=True))
    context, status_context = AsyncMock(), AsyncMock()
    context.__aenter__.return_value = session
    status_context.__aenter__.return_value = marker
    monkeypatch.setattr(live, 'get_db_session', Mock(side_effect=[context, status_context, status_context]))
    return session, marker


def test_locked_product_does_not_fetch_or_write(monkeypatch):
    session, marker = storage(monkeypatch)
    session.execute.return_value = Mock(scalar_one=Mock(return_value=False))
    fetch = Mock(side_effect=AssertionError('must not fetch'))
    monkeypatch.setitem(live.LIVE_LOADERS, 'swpc.rtsw_mag', fetch)
    result = asyncio.run(live.collect_product('swpc.rtsw_mag', {}, NOW-timedelta(days=1), NOW))
    assert result.empty
    fetch.assert_not_called()
    marker.execute.assert_not_awaited()
    session.commit.assert_not_awaited()


def test_collection_writes_full_feed_once_and_reports_only_usable_selected_values(monkeypatch):
    session, marker = storage(monkeypatch)
    fetch = Mock(return_value=[dict(observed_at=NOW, received_at=NOW, active=True,
                                   values={'bx': 1., 'by': None, 'bz': -4., 'bt': 6.}, raw={})])
    monkeypatch.setattr(loaders.solar_wind_loader, 'fetch_records', fetch)
    report = asyncio.run(live.collect_live(load_observation_config(), ['bz', 'by'], now=NOW))
    fetch.assert_called_once_with('mag')
    assert report['downloaded_measurements'] == 1
    assert report['failed_metrics'] == ['by']
    # Advisory lock, one measurement insert, then atomic status update; no SELECT of observations.
    assert session.execute.await_count == 3
    insert = session.execute.call_args_list[1].args[0].compile(dialect=postgresql.dialect())
    assert [insert.params[f'metric_m{i}'] for i in range(4)] == ['bx', 'by', 'bz', 'bt']
    assert insert.params['value_m1'] is None
    assert insert.params['quality_m1'] == 'missing'
    assert insert.params['received_at_m2'] == NOW
    status = session.execute.call_args_list[2].args[0].compile(dialect=postgresql.dialect())
    assert status.params['data_quality'] == 'partial'
    session.commit.assert_awaited_once()
    marker.commit.assert_awaited_once()


def test_storage_failure_rolls_back_records_and_reports_failure(monkeypatch):
    from sqlalchemy.exc import SQLAlchemyError
    session, marker = storage(monkeypatch)
    session.execute.side_effect = [Mock(scalar_one=Mock(return_value=True)), SQLAlchemyError('write failed')]
    monkeypatch.setattr(loaders.geomagnetic_loader, 'fetch_records', Mock(return_value=[dict(
        interval_start=NOW, value=-20., quality='unverified', received_at=NOW, raw={})]))
    report = asyncio.run(live.collect_live(load_observation_config(), ['dst'], now=NOW))
    assert report['failed_metrics'] == ['dst']
    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()
    assert marker.commit.await_count == 2
    failure = marker.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    assert failure.params['last_error_code'] == 'storage_error'


def test_flagged_latest_values_are_saved_but_not_accepted_as_usable(monkeypatch):
    session, _ = storage(monkeypatch)
    monkeypatch.setattr(loaders.geomagnetic_loader, 'fetch_records', Mock(return_value=[dict(
        interval_start=NOW, value=2., quality='flagged', received_at=NOW, raw={'a_running': 8})]))
    report = asyncio.run(live.collect_live(load_observation_config(), ['kp'], now=NOW))
    assert report['downloaded_measurements'] == 0
    assert report['failed_metrics'] == ['kp']
    insert = session.execute.call_args_list[1].args[0].compile(dialect=postgresql.dialect())
    assert insert.params['quality_m0'] == 'flagged'
    assert insert.params['metric_m1'] == 'ap'
    status = session.execute.call_args_list[2].args[0].compile(dialect=postgresql.dialect())
    assert status.params['data_quality'] == 'unavailable'


def test_f107_uses_common_persistence_and_keeps_last_duplicate(monkeypatch):
    session, _ = storage(monkeypatch)
    monkeypatch.setattr(loaders.f107, 'fetch_f107', Mock(return_value=pd.DataFrame({
        'issue_time': [NOW, NOW, NOW + timedelta(days=1)], 'f10_7': [150., 160., 999.]})))
    report = asyncio.run(live.collect_live(load_observation_config(), ['f10_7'], now=NOW))
    assert report['failed_metrics'] == []
    assert report['downloaded_measurements'] == 1
    session.execute.assert_awaited_once()
    insert = session.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    assert insert.params['value_m0'] == 160.
    assert insert.params['source_product_m0'] == 'swpc.f107'
    assert 'quality IS DISTINCT FROM' in str(insert)
    session.commit.assert_awaited_once()
