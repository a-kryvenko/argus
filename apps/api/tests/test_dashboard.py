from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dashboard_auth import hash_password, verify_password, current_user
from app.db.session import get_db_session
from app.routers.dashboard import router
from app.services.api_statistics import summarize, record


def test_password_hashes_are_salted_and_verified():
    one = hash_password('correct horse battery staple')
    two = hash_password('correct horse battery staple')
    assert one != two
    assert verify_password('correct horse battery staple', one)
    assert not verify_password('incorrect password', one)
    assert not verify_password('anything', 'malformed')


def test_anonymous_cannot_access_dashboard():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db_session] = lambda: AsyncMock()
    with TestClient(app) as client:
        for path in ['/me', '/users', '/groups', '/observations', '/observations?kind=normalized', '/api-stats']:
            assert client.get('/dashboard'+path).status_code == 401
        assert client.post('/dashboard/users', headers={'Origin': 'http://localhost:3000'},
                           json={'username': 'test', 'password': 'long-password'}).status_code == 401


def test_writes_reject_missing_or_foreign_origin():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db_session] = lambda: AsyncMock()
    with TestClient(app) as client:
        for headers in [{}, {'Origin': 'https://evil.example'}, {'Origin': 'null'}]:
            for endpoint in ['/login', '/logout', '/users']:
                assert client.post('/dashboard'+endpoint, headers=headers, json={}).status_code == 403
            assert client.patch('/dashboard/users/1', headers=headers, json={}).status_code == 403


def test_user_without_permissions_is_forbidden():
    app = FastAPI()
    app.include_router(router)
    db = AsyncMock()
    from unittest.mock import Mock
    db.scalars.return_value = Mock(all=Mock(return_value=[]))
    app.dependency_overrides[get_db_session] = lambda: db
    app.dependency_overrides[current_user] = lambda: SimpleNamespace(id=1, username='reader', active=True)
    with TestClient(app) as client:
        assert client.get('/dashboard/me').status_code == 200
        for endpoint in ['/users', '/groups', '/observations', '/api-stats']:
            assert client.get('/dashboard'+endpoint).status_code == 403


def test_statistics_weighted_mean_statuses_and_p95():
    hour = datetime(2026, 9, 11, tzinfo=timezone.utc)
    def row(count, duration, status, bucket):
        return SimpleNamespace(hour=hour, route='/users/{user_id}', method='GET',
            count=count, duration_ms=duration, status=status, bucket_ms=bucket)
    result = summarize([row(94, 940, 200, 10), row(5, 500, 404, 100), row(1, 90000, 500, -1)])
    assert result['summary'] == {'requests': 100, 'errors_4xx': 5, 'errors_5xx': 1, 'average_ms': 914.4, 'p95_upper_ms': 100}
    assert len(result['routes']) == len(result['hours']) == 1
    assert summarize([])['summary']['p95_upper_ms'] is None
    assert summarize([row(1, 90000, 500, -1)])['summary']['p95_upper_ms'] == -1


def test_metric_cardinality_is_bounded(monkeypatch):
    from app.services import api_statistics
    monkeypatch.setattr(api_statistics, '_buffer', {})
    record('/users/{user_id}', 'UNRECOGNIZED', 200, 12)
    key = next(iter(api_statistics._buffer))
    assert key[1:] == ('/users/{user_id}', 'OTHER', 200, 25)


def test_empty_origin_configuration_fails_closed(monkeypatch):
    monkeypatch.setenv('DASHBOARD_ORIGINS', '')
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db_session] = lambda: AsyncMock()
    with TestClient(app) as client:
        assert client.post('/dashboard/logout').status_code == 403
