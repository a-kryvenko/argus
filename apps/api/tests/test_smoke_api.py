from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from api.db.session import get_request_db_session
from api.main import app


def test_dashboard_does_not_expose_public_routes(monkeypatch):
    async def unused_session():
        yield None

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
