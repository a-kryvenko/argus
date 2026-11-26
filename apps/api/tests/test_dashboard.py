import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from api.services.authentication import Principal, current_user, hash_password, verify_password
from api.db.session import get_request_db_session
from api.routers.dashboard import router, users
from api.schemas.dashboard import UserUpdate
from api.services.dashboard_users import update_user


def test_password_hashes_are_salted_and_verified():
    one = hash_password('correct horse battery staple')
    two = hash_password('correct horse battery staple')
    assert one != two
    assert verify_password('correct horse battery staple', one)
    assert not verify_password('incorrect password', one)
    assert not verify_password('anything', 'malformed')


def test_identity_lookup_releases_the_transaction_before_the_handler():
    request = Request({'type': 'http', 'headers': [(b'cookie', b'argus_session=test-session')]})
    db = AsyncMock()
    db.execute.return_value = Mock(one_or_none=Mock(return_value=(1, 'reader', True)))
    assert asyncio.run(current_user(request, db)) == Principal(1, 'reader', True)
    db.rollback.assert_awaited_once()


def test_reapplying_the_same_memberships_keeps_sessions():
    db = AsyncMock()
    db.scalar.return_value = Principal(2, 'reader', True)
    db.scalars.return_value = Mock(all=Mock(return_value=['readers']))
    db.execute.return_value = [(2, SimpleNamespace(name='readers', permissions=['read']))]
    result = asyncio.run(update_user(
        db, Principal(1, 'admin', True), 2, UserUpdate(groups=['readers', 'readers']), None))
    assert result['groups'] == ['readers']
    db.add_all.assert_not_called()
    assert all(call.args[0].is_select for call in db.execute.call_args_list)
    db.commit.assert_awaited_once()


def test_users_page_loads_all_groups_in_one_query():
    db = AsyncMock()
    page = [SimpleNamespace(id=i, username=f'user{i}', active=True) for i in range(50)]
    db.scalars.return_value = Mock(all=Mock(return_value=page))
    db.execute.return_value = [(0, SimpleNamespace(name='admins', permissions=['write', 'read'])),
                               (0, SimpleNamespace(name='readers', permissions=['read'])),
                               (1, SimpleNamespace(name='readers', permissions=['read']))]
    db.scalar.return_value = 100
    data = asyncio.run(users(page=1, db=db)).data
    assert data['total'] == 100 and len(data['items']) == 50
    assert data['items'][0]['groups'] == ['admins', 'readers']
    assert data['items'][0]['permissions'] == ['read', 'write']
    assert data['items'][1]['permissions'] == ['read']
    assert all(item['groups'] == item['permissions'] == [] for item in data['items'][2:])
    db.execute.assert_awaited_once()
    db.scalars.assert_awaited_once()
    db.scalar.assert_awaited_once()


def test_anonymous_cannot_access_dashboard():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_request_db_session] = lambda: AsyncMock()
    with TestClient(app) as client:
        for path in ['/me', '/users', '/groups', '/project-monitoring']:
            assert client.get('/dashboard'+path).status_code == 401
        assert client.post('/dashboard/users', headers={'Origin': 'http://localhost:3000'},
                           json={'username': 'test', 'password': 'long-password'}).status_code == 401


def test_writes_reject_missing_or_foreign_origin():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_request_db_session] = lambda: AsyncMock()
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
    app.dependency_overrides[get_request_db_session] = lambda: db
    app.dependency_overrides[current_user] = lambda: SimpleNamespace(id=1, username='reader', active=True)
    with TestClient(app) as client:
        assert client.get('/dashboard/me').status_code == 200
        for endpoint in ['/users', '/groups', '/project-monitoring']:
            assert client.get('/dashboard'+endpoint).status_code == 403


def test_empty_origin_configuration_fails_closed(monkeypatch):
    monkeypatch.setenv('DASHBOARD_ORIGINS', '')
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_request_db_session] = lambda: AsyncMock()
    with TestClient(app) as client:
        assert client.post('/dashboard/logout').status_code == 403


def test_dashboard_origin_is_shared_with_the_application(monkeypatch):
    monkeypatch.setenv('DASHBOARD_ORIGINS', 'https://old.example')
    app = FastAPI()
    app.state.dashboard_origins = frozenset({'https://dashboard.example'})
    app.include_router(router)
    app.dependency_overrides[get_request_db_session] = lambda: AsyncMock()
    with TestClient(app) as client:
        assert client.post('/dashboard/logout', headers={'Origin': 'https://old.example'}).status_code == 403
        result = client.post('/dashboard/logout', headers={'Origin': 'https://dashboard.example'})
        assert result.status_code == 200
        cookie = result.headers['set-cookie']
        assert 'Max-Age=0' in cookie and 'HttpOnly' in cookie and 'SameSite=strict' in cookie
