import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from sqlalchemy.dialects import postgresql

from clio.config import load_observation_config
from clio.ingestion import live_adapters as adapters
from clio.observations import live
from clio.observations.store import upsert_measurements

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def frame(*rows):
    return pd.DataFrame(rows, columns=['metric', 'value', 'observed_at']).assign(received_at=NOW)


def policies(**sources):
    return {metric: SimpleNamespace(sources=SimpleNamespace(live=names)) for metric, names in sources.items()}


def test_one_fetch_for_magnetic_and_plasma_with_separate_provenance():
    wind = SimpleNamespace(fetch=AsyncMock(return_value=frame(('bx', -3., NOW), ('v', 450., NOW))))
    result, report = asyncio.run(live.fetch_live(['bx', 'v'], policies(bx=['mag'], v=['plasma']),
                                                {'mag': wind, 'plasma': wind}, NOW))
    wind.fetch.assert_awaited_once()
    assert result.source_product.tolist() == ['mag', 'plasma']
    assert report['failed_metrics'] == []


def test_fallback_fills_missing_slots_without_replacing_primary_values():
    primary = SimpleNamespace(fetch=AsyncMock(return_value=frame(('v', 400., NOW - timedelta(hours=1)),
                                                                 ('n', 5., NOW))))
    fallback = SimpleNamespace(fetch=AsyncMock(return_value=frame(('v', 450., NOW), ('n', 99., NOW))))
    result, report = asyncio.run(live.fetch_live(['v', 'n'], policies(v=['p', 'f'], n=['p', 'f']),
                                                {'p': primary, 'f': fallback}, NOW))
    assert result[result.metric == 'n'].value.tolist() == [5.]
    assert result[result.metric == 'v'].value.tolist() == [400., 450.]
    assert report['failed_metrics'] == []


def test_shared_failure_is_not_fetched_twice_and_other_sources_succeed():
    failed = SimpleNamespace(fetch=AsyncMock(side_effect=OSError('offline')))
    good = SimpleNamespace(fetch=AsyncMock(return_value=frame(('dst', -70., NOW))))
    result, report = asyncio.run(live.fetch_live(['bx', 'v', 'dst'], policies(bx=['mag'], v=['plasma'], dst=['dst']),
                                                {'mag': failed, 'plasma': failed, 'dst': good}, NOW))
    failed.fetch.assert_awaited_once()
    assert result.value.tolist() == [-70.]
    assert set(report['failed_metrics']) == {'bx', 'v'}
    assert [a['error'] for a in report['source_attempts'] if 'error' in a] == ['offline', 'offline']


def test_freshness_quality_and_future_times_are_checked_per_metric():
    source = SimpleNamespace(fetch=AsyncMock(return_value=frame(
        ('v', 400., NOW - timedelta(hours=1)), ('v', 500., NOW + timedelta(minutes=1)),
        ('n', -1e31, NOW), ('kp', 10., NOW), ('dst', -80., NOW - timedelta(hours=2)),
        ('f10_7', 150., NOW - timedelta(days=1)))))
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

        sources = {m: SimpleNamespace(fetch=lambda start, now, m=m: fetch(m, start, now)) for m in ('v', 'dst')}
        _, report = await live.fetch_live(['v', 'dst'], policies(v=['v'], dst=['dst']), sources, NOW)
        assert report['failed_metrics'] == []

    asyncio.run(scenario())


def test_native_adapter_preserves_fractional_kp_raw_ap_and_receipt(monkeypatch):
    received = NOW - timedelta(minutes=1)
    records = [dict(interval_start=NOW - timedelta(hours=3), quality='unverified', value=3.33,
                    raw={'a_running': 18.}, received_at=received),
               dict(interval_start=NOW, quality='flagged', value=5., raw={'a_running': 39.}, received_at=NOW)]
    ingest = AsyncMock(return_value=records)
    monkeypatch.setattr(adapters, 'ingest_source', ingest)
    result = asyncio.run(adapters.GeomagneticLiveAdapter('kp').fetch(NOW - timedelta(days=1), NOW))
    assert result.value.tolist() == [3.33, 18.]
    assert result.metric.tolist() == ['kp', 'ap']
    assert set(result.received_at) == {received}


def test_native_error_propagates(monkeypatch):
    monkeypatch.setattr(adapters, 'ingest_source', AsyncMock(side_effect=OSError('offline')))
    with pytest.raises(OSError):
        asyncio.run(adapters.GeomagneticLiveAdapter('dst').fetch(NOW - timedelta(days=1), NOW))


def test_live_saves_partial_results_with_priority_policy_and_no_normalization(monkeypatch):
    config = load_observation_config()
    wind = SimpleNamespace(fetch=AsyncMock(return_value=frame(('v', 450., NOW))))
    monkeypatch.setattr(live, 'live_adapters', lambda *_: {'swpc.propagated_plasma': wind})
    save = AsyncMock()
    monkeypatch.setattr(live, 'upsert_measurements', save)
    session = AsyncMock()
    report = asyncio.run(live.collect_live(session, config, ['v', 'n'], now=NOW))
    assert report['failed_metrics'] == ['n']
    assert report['downloaded_measurements'] == 1
    assert save.call_args.kwargs == {'source_priorities': {'v': ['swpc.propagated_plasma'], 'n': ['swpc.propagated_plasma']}}
    session.commit.assert_awaited_once()


def test_live_noop_does_not_refresh_receipt_and_update_requires_priority():
    session = AsyncMock()
    session.execute.return_value = Mock(all=lambda: [])
    records = frame(('v', 450., NOW)).assign(source_product='backup')
    asyncio.run(upsert_measurements(session, records, source_priorities={'v': ['primary', 'backup']}))
    session.execute.assert_awaited_once()
    compiled = session.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert 'RETURNING' in sql and 'source_product IS DISTINCT FROM' in sql
    # A backup may correct its own observations, but not a known primary value.
    lists = [value for value in compiled.params.values() if isinstance(value, list)]
    assert ['backup'] in lists and ['primary', 'backup'] in lists


def test_live_receipt_uses_actual_changed_record_time():
    session = AsyncMock()
    received = NOW - timedelta(seconds=20)
    session.execute.return_value = Mock(all=lambda: [('v', NOW - timedelta(minutes=1), received)])
    records = frame(('v', 450., NOW - timedelta(minutes=1))).assign(source_product='primary', received_at=received)
    asyncio.run(upsert_measurements(session, records, source_priorities={'v': ['primary']}))
    assert session.execute.await_count == 2
    receipt = session.execute.call_args_list[1].args[0].compile(dialect=postgresql.dialect())
    assert receipt.params['received_at'] == received
