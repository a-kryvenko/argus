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
            assert client.get(path).status_code == 404
        assert client.post('/risks/leo-drag', json={}).status_code == 404
