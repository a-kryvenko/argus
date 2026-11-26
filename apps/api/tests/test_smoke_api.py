from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from api.db.session import get_request_db_session
from api.main import app, create_app


def test_dashboard_does_not_expose_public_routes(monkeypatch):
    async def unused_session():
        yield SimpleNamespace(rollback=AsyncMock())

    monkeypatch.setitem(app.dependency_overrides, get_request_db_session, unused_session)
    with TestClient(app) as client:
        assert client.get('/api/v1/dashboard/me').status_code == 401
        for path in ('/ping', '/observations', '/observations/history', '/summary',
                     '/forecasts/dst', '/docs', '/openapi.json'):
            assert client.get('/api/v1' + path).status_code == 404
        assert client.post('/api/v1/risks/leo-drag', json={}).status_code == 404


def test_dashboard_requires_versioned_paths(monkeypatch):
    @asynccontextmanager
    async def ready_session():
        yield SimpleNamespace(execute=AsyncMock())

    monkeypatch.setattr('api.routers.healthcheck.get_db_session', ready_session)
    with TestClient(app) as client:
        for path in ('/health/live', '/health/ready', '/dashboard/me'):
            assert client.get(path).status_code == 404
        for path in ('/api/v1/health/live', '/api/v1/health/ready'):
            response = client.get(path)
            assert response.status_code == 200
            assert response.json() == {'service': 'api', 'status': 'ok'}


def test_dashboard_headers_use_configured_origin_and_preserve_http_headers(monkeypatch):
    origin = 'https://dashboard.test'
    monkeypatch.setenv('DASHBOARD_ORIGINS', origin)
    application = create_app()
    with TestClient(application) as client:
        path = '/api/v1/dashboard/me'
        preflight = client.options(path, headers={
            'Origin': origin, 'Access-Control-Request-Method': 'GET',
        })
        assert preflight.status_code == 200
        assert preflight.headers['Access-Control-Allow-Origin'] == origin
        assert preflight.headers['Access-Control-Allow-Credentials'] == 'true'
        response = client.post(path, headers={'Origin': origin})
        assert response.status_code == 405
        assert response.headers['Allow'] == 'GET'
        assert response.headers['Cache-Control'] == 'no-store'
        assert response.headers['Access-Control-Allow-Origin'] == origin


def test_dashboard_errors_follow_the_application_contract(monkeypatch):
    origin = 'https://dashboard.test'
    monkeypatch.setenv('DASHBOARD_ORIGINS', origin)
    monkeypatch.setattr('api.main.get_config', lambda: SimpleNamespace(debug=False))
    application = create_app()

    async def unused_session():
        yield SimpleNamespace(rollback=AsyncMock())

    async def broken_session():
        raise RuntimeError('Private diagnostic')

    application.dependency_overrides[get_request_db_session] = unused_session
    with TestClient(application, headers={'Origin': origin}) as client:
        response = client.post('/api/v1/dashboard/login', json={'username': ['invalid'], 'password': 'example'})
        assert response.status_code == 422
        assert response.json()['error']['code'] == 'INVALID_INPUT'
        assert 'example' not in response.text
        assert response.headers['Cache-Control'] == 'no-store'

        application.dependency_overrides[get_request_db_session] = broken_session
        response = client.get('/api/v1/dashboard/me')
        assert response.status_code == 500
        assert response.json()['error']['code'] == 'INTERNAL_ERROR'
        assert 'Private diagnostic' not in response.text
        assert response.headers['Cache-Control'] == 'no-store'
        assert response.headers['Access-Control-Allow-Origin'] == origin
