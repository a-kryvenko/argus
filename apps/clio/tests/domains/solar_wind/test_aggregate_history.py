import asyncio
import pytest
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from clio.domains.solar_wind.history import history
from clio.routers import solar_wind as routes

START = datetime(2026, 9, 1, tzinfo=UTC)


def test_statistics_include_missing_windows_in_coverage():
    row = dict(metric='bz', bucket=START, received_at=START, samples=4, count=3,
               mean=-2, min=-8, max=3, negative_count=2, spacecraft='A', last_spacecraft='B', source_changes=1)
    session = AsyncMock()
    session.execute.return_value = Mock(mappings=lambda: Mock(all=lambda: [row]))
    data = asyncio.run(history(session, ['bz'], START, START+timedelta(minutes=10), 300))
    series = data['series']['bz']
    assert series['points'][0]['min'] == -8
    assert series['points'][0]['received_at'] == START
    assert series['points'][0]['source_changes'] == 1
    assert series['coverage']['percent'] == 30
    assert series['coverage']['expected_slots'] == 10
    assert series['coverage']['missing_slots'] == 6
    assert series['coverage']['invalid_slots'] == 1
    assert [gap['reason'] for gap in series['coverage']['gaps']] == ['partial', 'missing']
    assert 'processing' not in series
    assert 'clio.measurement' in str(session.execute.call_args.args[0])


def test_partial_current_bucket_excluded_and_empty_history_is_missing():
    session = AsyncMock()
    session.execute.return_value = Mock(mappings=lambda: Mock(all=lambda: []))
    data = asyncio.run(history(session, ['bz'], START+timedelta(minutes=2), START+timedelta(minutes=8), 300, now=START+timedelta(minutes=7)))
    assert data['evaluated_from'] == START
    assert data['evaluated_to'] == START+timedelta(minutes=5)
    assert data['series']['bz']['coverage']['expected_slots'] == 5
    assert data['series']['bz']['coverage']['percent'] == 0
    assert data['series']['bz']['coverage']['gaps'][0]['slots'] == 5


def test_resolution_routing_and_bounds(monkeypatch):
    app = FastAPI(); app.include_router(routes.router)
    app.dependency_overrides[routes.get_db_session] = lambda: object()
    aggregate = AsyncMock(return_value={})
    monkeypatch.setattr(routes.aggregate_history, 'history', aggregate)
    with TestClient(app) as client:
        base = '/internal/v1/observations/solar-wind/history?from=2026-08-01T00:00:00Z&to=2026-08-31T00:00:00Z'
        assert client.get(base).status_code == 422
        assert client.get(base+'&resolution=auto').status_code == 200
        assert aggregate.call_args.args[-1] == 3600
        assert client.get(base+'&resolution=5m').status_code == 200
        assert aggregate.call_args.args[-1] == 300
        assert client.get(base+'&resolution=bad').status_code == 422


@pytest.mark.parametrize('hours,seconds', [(6, None), (6.1, 300), (24, 300), (24.1, 3600), (72, 3600), (168, 3600), (720, 3600)])
def test_auto_resolution_bounds_keep_daily_histories_compact(monkeypatch, hours, seconds):
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_db_session] = lambda: object()
    native = AsyncMock(return_value={'series': {}})
    aggregate = AsyncMock(return_value={'series': {}})
    monkeypatch.setattr(routes.solar_wind, 'history', native)
    monkeypatch.setattr(routes.aggregate_history, 'history', aggregate)
    with TestClient(app) as client:
        response = client.get('/internal/v1/observations/solar-wind/history', params={
            'from': START.isoformat(), 'to': (START+timedelta(hours=hours)).isoformat(), 'resolution': 'auto',
        })
        assert response.status_code == 200
        assert response.headers['cache-control'] == 'no-store'
        if seconds is None:
            native.assert_awaited_once()
            aggregate.assert_not_awaited()
        else:
            assert aggregate.await_args.args[-1] == seconds
            native.assert_not_awaited()
