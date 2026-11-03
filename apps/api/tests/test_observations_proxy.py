import asyncio
from unittest.mock import AsyncMock
import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.services.observations import client as service
from api.routers import observations
from api.routers import dashboard


def test_public_routes_use_clio_without_database_dependency(monkeypatch):
    read = AsyncMock(return_value={'success': True, 'data': {'points': []}, 'error': None})
    monkeypatch.setattr(observations, 'read_observations', read)
    app = FastAPI()
    app.include_router(observations.router)
    with TestClient(app) as client:
        assert client.get('/observations/history?observations=ap&limit=12').json()['data']['points'] == []
        read.assert_awaited_once_with('history', {'limit': 12})
        assert client.get('/observations/history?limit=169').status_code == 422


def test_dashboard_observation_browser_is_removed():
    app = FastAPI()
    app.include_router(dashboard.router)
    with TestClient(app) as client:
        assert client.get('/dashboard/observations').status_code == 404
        assert client.get('/dashboard/observations?kind=normalized').status_code == 404


@pytest.mark.parametrize('status', [401, 403, 500, 503])
def test_owner_failure_maps_to_503_without_sql_fallback(monkeypatch, status):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(status)))
    monkeypatch.setenv('OBSERVATIONS_URL', 'http://clio')
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'secret')
    monkeypatch.setattr(service.httpx, 'AsyncClient', lambda **_: client)
    with pytest.raises(HTTPException) as error:
        asyncio.run(service.read_observations('history'))
    assert error.value.status_code == 503


def test_versioned_read_path_and_authentication(monkeypatch):
    def response(request):
        assert request.url.path == '/internal/v1/observations/history'
        assert request.url.params['limit'] == '2'
        assert request.headers['authorization'] == 'Bearer secret'
        return httpx.Response(200, json={'success': True, 'data': {'items': []}, 'error': None})
    client = httpx.AsyncClient(transport=httpx.MockTransport(response))
    monkeypatch.setenv('OBSERVATIONS_URL', 'http://clio')
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'secret')
    monkeypatch.setattr(service.httpx, 'AsyncClient', lambda **_: client)
    assert asyncio.run(service.read_observations('history', {'limit': 2}))['data']['items'] == []


def test_validation_response_is_preserved(monkeypatch):
    payload = {'detail': [{'msg': 'Invalid time range'}]}
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(422, json=payload)))
    monkeypatch.setenv('OBSERVATIONS_URL', 'http://clio')
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'secret')
    monkeypatch.setattr(service.httpx, 'AsyncClient', lambda **_: client)
    result = asyncio.run(service.read_observations('history'))
    assert result.status_code == 422


def test_hourly_history_filters_fields_and_passes_time_bounds(monkeypatch):
    from datetime import UTC, datetime
    read = AsyncMock(return_value={'success': True, 'data': {'points': [
        {'issue_time': '2026-09-01T12:00:00Z', 'ap': 4, 'f10_7': 120, 's10': None},
    ]}})
    monkeypatch.setattr(observations, 'read_observations', read)
    app = FastAPI()
    app.include_router(observations.router)
    with TestClient(app) as client:
        result = client.get('/observations/history', params={
            'observations': 'ap,s10,ap', 'limit': 12,
            'from': '2026-09-01T00:00:00Z', 'to': '2026-09-02T00:00:00Z',
        })
        assert result.status_code == 200
        assert result.json()['data']['points'] == [
            {'issue_time': '2026-09-01T12:00:00Z', 'ap': 4, 's10': None}]
        read.assert_awaited_once_with('history', {'limit': 12,
            'from': datetime(2026, 9, 1, tzinfo=UTC), 'to': datetime(2026, 9, 2, tzinfo=UTC)})
        for params in [
            {'from': '2026-09-01T00:00:00', 'to': '2026-09-02T00:00:00Z'},
            {'from': '2026-09-03T00:00:00Z', 'to': '2026-09-02T00:00:00Z'},
        ]:
            assert client.get('/observations/history', params={'observations': 'ap', **params}).status_code == 422


def test_unified_history_preserves_native_validation_errors(monkeypatch):
    from fastapi.responses import JSONResponse
    monkeypatch.setattr(observations, 'read_observations', AsyncMock(return_value=
        JSONResponse(status_code=422, content={'detail': 'Invalid interval'})))
    app = FastAPI()
    app.include_router(observations.router)
    with TestClient(app) as client:
        result = client.get('/observations/history?observations=bz')
        assert result.status_code == 422
        assert result.json() == {'detail': 'Invalid interval'}


@pytest.mark.parametrize('selected,resolution,days', [
    ('bz', '1m', 8), ('bz', '5m', 32), ('bz', '1h', 367),
    ('kp', 'auto', 32), ('bz,kp', 'auto', 32),
])
def test_history_rejects_oversized_ranges_before_reading(monkeypatch, selected, resolution, days):
    from datetime import UTC, datetime, timedelta
    read = AsyncMock()
    monkeypatch.setattr(observations, 'read_observations', read)
    app = FastAPI()
    app.include_router(observations.router)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    with TestClient(app) as client:
        result = client.get('/observations/history', params={
            'observations': selected, 'resolution': resolution,
            'from': start.isoformat(), 'to': (start + timedelta(days=days)).isoformat(),
        })
        assert result.status_code == 422
    read.assert_not_awaited()


@pytest.mark.parametrize('hours,resolution', [(6, '1m'), (6.1, '5m'), (24, '5m'), (24.1, '1h'), (72, '1h'), (168, '1h'), (720, '1h')])
def test_history_selects_auto_resolution_and_passes_selected_metrics(monkeypatch, hours, resolution):
    from datetime import UTC, datetime, timedelta
    read = AsyncMock(return_value={'success': False, 'data': None,
        'error': {'code': 'NOT_READY', 'message': 'Not ready'}})
    monkeypatch.setattr(observations, 'read_observations', read)
    app = FastAPI()
    app.include_router(observations.router)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=hours)
    with TestClient(app) as client:
        result = client.get('/observations/history', params={
            'observations': 'bz,v', 'resolution': 'auto', 'from': start.isoformat(), 'to': end.isoformat(),
        })
        assert result.status_code == 200
        assert result.json()['error']['code'] == 'NOT_READY'
    read.assert_awaited_once_with('solar-wind/history', {
        'from': start, 'to': end, 'resolution': resolution, 'metrics': 'bz,v'})
