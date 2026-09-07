import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from app.db.models import SolarWindObservation
from app.routers.public import solar_wind as routes
from app.services import solar_wind as service

NOW = datetime(2026, 9, 7, 12, tzinfo=UTC)


def observation(kind='mag', minutes=1, **overrides):
    return SolarWindObservation(kind=kind, observed_at=NOW - timedelta(minutes=minutes),
                                received_at=NOW, spacecraft='SOLAR1', active=True,
                                values={'bz': -4, 'bt': 6, 'v': 420, 'n': None},
                                raw={'overall_quality': 0}, **overrides)


def test_latest_has_independent_freshness_and_does_not_hide_missing_values():
    session = AsyncMock()
    session.execute.side_effect = [Mock(scalars=Mock(return_value=Mock(first=Mock(return_value=observation())))),
                                   Mock(scalars=Mock(return_value=Mock(first=Mock(return_value=observation('plasma', 20)))))]
    result = asyncio.run(service.latest(session, ['bz', 'v', 'n'], NOW))['series']
    assert result['bz']['status'] == 'fresh'
    assert result['bz']['age_seconds'] == 60
    assert result['v']['status'] == 'stale'
    assert result['v']['age_seconds'] == 1200
    assert result['n']['status'] == 'missing'
    assert result['n']['latest']['value'] is None
    assert result['bz']['coordinate_system'] == 'GSM'
    assert result['bz']['propagated'] is False
    for call in session.execute.call_args_list:
        sql = str(call.args[0].compile(dialect=postgresql.dialect()))
        assert 'LIMIT' in sql and 'active IS true' in sql


def test_history_preserves_gaps_and_flagged_values():
    rows = [observation(minutes=20), observation(minutes=1)]
    rows[1].raw = {'overall_quality': 2}
    session = AsyncMock()
    session.execute.return_value = Mock(scalars=Mock(return_value=rows))
    result = asyncio.run(service.history(session, ['bz'], NOW-timedelta(hours=1), NOW))
    points = result['series']['bz']['points']
    assert len(points) == 2
    assert points[1]['quality'] == 'flagged'
    assert points[1]['value'] == -4
    assert result['gap_filling'] == 'none'
    assert points[1]['spacecraft'] == 'SOLAR1'
    sql = str(session.execute.call_args.args[0].compile(dialect=postgresql.dialect()))
    assert 'DISTINCT ON' in sql and 'active IS true' in sql


def test_empty_database_has_explicit_missing_series():
    session = AsyncMock()
    session.execute.return_value = Mock(scalars=Mock(return_value=Mock(first=Mock(return_value=None))))
    series = asyncio.run(service.latest(session, ['bz'], NOW))['series']['bz']
    assert series['latest'] is None and series['age_seconds'] is None
    assert series['status'] == 'missing'


def test_api_validates_ranges_and_metrics_without_querying_database(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router)
    session = object()
    app.dependency_overrides[routes.get_db_session] = lambda: session
    latest, history = AsyncMock(return_value={'series': {}}), AsyncMock(return_value={'series': {}})
    monkeypatch.setattr(routes.solar_wind, 'latest', latest)
    monkeypatch.setattr(routes.solar_wind, 'history', history)
    prefix = '/public/observations/solar-wind'
    with TestClient(app) as client:
        for query in ['metrics=unknown', 'metrics=', 'from=2026-09-01T00:00:00',
                      'from=2026-09-01T00:00:00Z&to=2026-09-09T00:00:00Z',
                      'from=2026-09-02T00:00:00Z&to=2026-09-01T00:00:00Z',
                      'from=bad']:
            assert client.get(f'{prefix}/history?{query}').status_code == 422
        history.assert_not_called()
        response = client.get(f'{prefix}/history?from=2026-09-01T00:00:00Z&to=2026-09-08T00:00:00Z&metrics=bz,bz,v')
        assert response.status_code == 200
        assert response.headers['cache-control'] == 'no-store'
        assert history.call_args.args[1] == ['bz', 'v']
        response = client.get(f'{prefix}/latest?metrics=bz')
        assert response.status_code == 200
        latest.assert_awaited_once_with(session, ['bz'])


def test_one_source_failure_does_not_cancel_other_ingestion(monkeypatch):
    completed = []
    async def ingest(kind):
        if kind == 'mag':
            raise OSError('provider unavailable')
        completed.append(kind)
    monkeypatch.setattr(service, 'ingest_source', ingest)
    with pytest.raises(RuntimeError, match='mag'):
        asyncio.run(service.refresh_solar_wind())
    assert completed == ['plasma']


def test_locked_collector_does_not_fetch_or_commit(monkeypatch):
    session = AsyncMock()
    session.execute.return_value = Mock(scalar_one=Mock(return_value=False))
    context = AsyncMock()
    context.__aenter__.return_value = session
    monkeypatch.setattr(service, 'get_session_factory', lambda: lambda: context)
    fetch = Mock(side_effect=AssertionError('must not fetch'))
    monkeypatch.setattr(service, 'fetch_records', fetch)
    asyncio.run(service.ingest_source('mag'))
    fetch.assert_not_called()
    session.commit.assert_not_awaited()
