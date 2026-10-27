"""All observation resources inherit authentication from the application."""
from unittest.mock import Mock

from fastapi.testclient import TestClient

from clio.db.session import get_request_db_session
from clio.main import app


def test_all_observation_routes_require_common_authentication(monkeypatch):
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'test-token')
    database = Mock(side_effect=AssertionError('Unauthenticated requests must not read storage'))
    monkeypatch.setitem(app.dependency_overrides, get_request_db_session, database)
    paths = [path for path in app.openapi()['paths'] if path.startswith('/internal/v1/observations')]
    assert paths
    client = TestClient(app)
    for path in paths:
        path = path.replace('{kind}', 'gong').replace('{sha256}', 'a'*64)
        assert client.get(path).status_code == 401
        assert client.get(path, headers={'Authorization': 'Bearer wrong'}).status_code == 401
    assert client.get('/health/live').status_code == 200
    database.assert_not_called()
