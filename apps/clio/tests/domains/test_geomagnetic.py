import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from clio.db.models import GeomagneticObservation
from clio.domains import geomagnetic as service
from clio.routers import geomagnetic as routes

NOW = datetime(2026, 9, 7, 16, tzinfo=UTC)


def record(metric, start, value):
    return GeomagneticObservation(metric=metric, interval_start=start,
        interval_end=start+timedelta(seconds=service.INTERVAL_SECONDS[metric]),
        value=value, quality='unverified' if value is not None else 'missing', received_at=NOW, raw={'station_count': 8})


def test_freshness_uses_end_of_each_native_interval():
    session = AsyncMock()
    kp = record('kp', NOW-timedelta(hours=1), 3.33)  # 15–18, in progress
    dst = record('dst', NOW-timedelta(hours=5), -45)
    session.execute.side_effect = [Mock(scalars=Mock(return_value=Mock(first=Mock(return_value=r)))) for r in [kp, dst]]
    series = asyncio.run(service.latest(session, NOW))['series']
    assert series['kp']['latest']['value'] == 3.33
    assert series['kp']['latest']['interval_status'] == 'in_progress'
    assert series['kp']['lag_seconds'] == 0 and series['kp']['status'] == 'fresh'
    assert series['dst']['lag_seconds'] == 14400 and series['dst']['status'] == 'stale'


def test_history_queries_overlapping_intervals_without_expanding_them():
    session = AsyncMock()
    session.execute.return_value = Mock(scalars=Mock(return_value=[record('kp', NOW-timedelta(hours=4), 3.33)]))
    result = asyncio.run(service.history(session, NOW-timedelta(hours=3), NOW, NOW))
    assert len(result['series']['kp']['points']) == 1
    assert result['series']['kp']['points'][0]['interval_start'] == NOW-timedelta(hours=4)
    sql = str(session.execute.call_args.args[0].compile(dialect=postgresql.dialect()))
    assert 'interval_end >' in sql and 'interval_start <' in sql


def test_empty_database_has_no_invented_indices():
    session = AsyncMock()
    session.execute.return_value = Mock(scalars=Mock(return_value=Mock(first=Mock(return_value=None))))
    series = asyncio.run(service.latest(session, NOW))['series']
    assert all(s['status'] == 'missing' and s['latest'] is None for s in series.values())


def test_independent_source_failure(monkeypatch):
    calls = []
    async def ingest(metric):
        calls.append(metric)
        if metric == 'kp':
            raise OSError('offline')
    monkeypatch.setattr(service, 'ingest_source', ingest)
    with pytest.raises(RuntimeError, match='kp'):
        asyncio.run(service.refresh_geomagnetic())
    assert calls == ['kp', 'dst']




def test_api_rejects_invalid_ranges_before_querying(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_db_session] = lambda: object()
    history = AsyncMock(return_value={'series': {}})
    monkeypatch.setattr(service, 'history', history)
    with TestClient(app) as client:
        prefix = '/internal/v1/observations/geomagnetic/history?'
        for query in ['from=bad', 'from=2026-09-07T00:00:00',
                      'from=2026-09-07T00:00:00Z&to=2026-09-06T00:00:00Z',
                      'from=2026-07-01T00:00:00Z&to=2026-09-01T00:00:00Z']:
            assert client.get(prefix+query).status_code == 422
        history.assert_not_called()
        response = client.get(prefix+'from=2026-09-07T13:00:00Z&to=2026-09-07T16:00:00Z')
        assert response.status_code == 200 and response.headers['cache-control'] == 'no-store'


def test_completed_kp_waits_for_next_native_interval_before_becoming_stale():
    session = AsyncMock()
    kp = record('kp', NOW-timedelta(hours=7), 3.33)  # 09–12, next interval + 1h grace
    def results():
        return [Mock(scalars=Mock(return_value=Mock(first=Mock(return_value=r)))) for r in [kp, None]]
    session.execute.side_effect = results()
    assert asyncio.run(service.latest(session, NOW))['series']['kp']['status'] == 'fresh'
    session.execute.side_effect = results()
    assert asyncio.run(service.latest(session, NOW+timedelta(seconds=1)))['series']['kp']['status'] == 'stale'


def test_stored_kp_projection_preserves_intervals_and_raw_ap():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    import pandas as pd
    from clio.domains.geomagnetic import load_kp_measurements

    clock = pd.date_range('2026-09-01', periods=5, freq='3h', tz='UTC')
    records = [SimpleNamespace(interval_start=stamp, value=kp, raw=raw)
               for stamp, kp, raw in zip(clock, [2.333, 3., None, 1., 4.],
                   [{'a_running': '12'}, {}, {'a_running': 7}, {'a_running': 'bad'}, {'a_running': float('nan')}])]
    result = Mock()
    result.all.return_value = records
    session = SimpleNamespace(scalars=AsyncMock(return_value=result))
    frame = asyncio.run(load_kp_measurements(session, since=clock[0], until=clock[-1]))
    assert frame[frame.metric == 'kp'].value.tolist() == [2.333, 3., 1., 4.]
    assert frame[frame.metric == 'ap'].value.tolist() == [12., 7.]
    assert frame[frame.metric == 'ap'].observed_at.tolist() == [clock[0], clock[2]]
    assert frame[frame.metric == 'kp'].observed_at.tolist() == list(clock[[0, 1, 3, 4]])
