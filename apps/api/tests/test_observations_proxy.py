import asyncio
from unittest.mock import AsyncMock
import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.services import observations_client as service
from app.routers.public import observations, solar_wind
from app.routers import dashboard


def test_public_routes_use_clio_without_database_dependency(monkeypatch):
    read = AsyncMock(return_value={'success': True, 'data': {'points': []}, 'error': None})
    monkeypatch.setattr(observations, 'read_observations', read)
    app = FastAPI()
    app.include_router(observations.router)
    with TestClient(app) as client:
        assert client.get('/public/observations/history?limit=12').json()['data'] == {'points': []}
        read.assert_awaited_once_with('history', {'limit': 12})
        assert client.get('/public/observations/history?limit=169').status_code == 422


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
        asyncio.run(service.read_observations('latest'))
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
