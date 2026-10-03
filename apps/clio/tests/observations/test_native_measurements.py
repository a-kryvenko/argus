import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

from sqlalchemy.dialects import postgresql

from clio.observations.native import wind_measurements, geomagnetic_measurements, store_native
from clio.monitoring import monitoring
from clio.db.models import Measurement

NOW = datetime(2026, 10, 3, tzinfo=UTC)


def wind(source, value, active=True):
    return dict(observed_at=NOW, received_at=NOW, spacecraft=source, active=active,
                values={'bz': value}, raw={'overall_quality': 0})


def test_wind_selects_one_active_spacecraft_and_preserves_missing_metrics():
    rows = wind_measurements('mag', [wind('B', -3), wind('A', -2), wind('C', 99, False)])
    assert len(rows) == 4
    assert {row['spacecraft'] for row in rows} == {'A'}
    assert next(row for row in rows if row['metric'] == 'bz')['value'] == -2
    assert next(row for row in rows if row['metric'] == 'bx')['quality'] == 'missing'
    assert wind_measurements('mag', [wind('C', 99, False)]) == []


def test_geomagnetic_ap_is_a_measurement_with_the_native_interval():
    point = dict(interval_start=NOW, interval_end=NOW+timedelta(hours=3), value=3.33,
                 quality='unverified', received_at=NOW, raw={'a_running': '18', 'station_count': 8})
    rows = geomagnetic_measurements('kp', [point])
    assert [(r['metric'], r['value']) for r in rows] == [('kp', 3.33), ('ap', 18.)]
    assert all(r['interval_end'] == NOW+timedelta(hours=3) for r in rows)
    point['raw']['a_running'] = float('inf')
    assert geomagnetic_measurements('kp', [point])[1]['quality'] == 'missing'


def test_upsert_updates_quality_but_receipt_alone_is_not_a_revision():
    session = AsyncMock()
    asyncio.run(store_native(session, wind_measurements('mag', [wind('A', -2)])))
    query = str(session.execute.call_args.args[0].compile(dialect=postgresql.dialect()))
    where = query.split(' WHERE ')[1]
    assert 'quality IS DISTINCT FROM' in where
    assert 'received_at IS DISTINCT FROM' not in where
    assert 'ON CONFLICT ON CONSTRAINT uq_measurement_metric' in query


def test_monitoring_reads_latest_measurement_and_its_receipt(monkeypatch):
    point = Measurement(metric='bz', value=-2, observed_at=NOW, received_at=NOW-timedelta(seconds=5))
    session = AsyncMock()
    session.scalar.return_value = point
    session.get.return_value = None
    monkeypatch.setattr(monitoring, 'OBSERVATION_METRICS', ['bz'])
    monkeypatch.setattr(monitoring, 'source_status', AsyncMock(return_value={'status': 'ok'}))
    result = asyncio.run(monitoring.monitoring_status(session))
    assert result['measurements'][0]['latest_observation_at'] == NOW
    assert result['measurements'][0]['received_at'] == point.received_at
    assert 'clio.measurement' in str(session.scalar.call_args.args[0])


def test_missing_latest_measurement_is_not_reported_as_fresh(monkeypatch):
    point = Measurement(metric='bz', value=None, observed_at=NOW, received_at=NOW, quality='missing')
    session = AsyncMock()
    session.scalar.return_value = point
    session.get.return_value = None
    monkeypatch.setattr(monitoring, 'OBSERVATION_METRICS', ['bz'])
    monkeypatch.setattr(monitoring, 'source_status', AsyncMock(return_value={'status': 'ok'}))
    result = asyncio.run(monitoring.monitoring_status(session))
    assert result['measurements'][0]['status'] == 'unavailable'
    assert result['status'] == 'degraded'
