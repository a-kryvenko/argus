"""Dashboard end-to-end API checks in a disposable PostgreSQL schema.

apps/api/.venv/bin/python apps/api/tests/integration/verify_dashboard.py
"""
import asyncio
from contextlib import contextmanager
import os
import socket
import subprocess
import time
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from api.db.session import get_db_session
from api.db.models.dashboard import Session, LoginAttempt
from api.services.authentication import digest
from api.routers.dashboard import router, create_user, UserCreate
from api.routers.observations import router as public_router
from api.services.monitoring import statistics as api_statistics

@contextmanager
def clio_server(root, schema):
    """Run the observation owner in its own installed environment over real HTTP."""
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    code = """
import asyncio
import os
from datetime import datetime, timedelta, timezone
import uvicorn
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from clio.main import app
from clio.db.session import get_db_session
from clio.db.models.measurement import Measurement
url = make_url(os.environ['TEST_DATABASE_ADMIN_DSN']).set(drivername='postgresql+psycopg')
engine = create_async_engine(url, execution_options={'schema_translate_map': {'clio': os.environ['TEST_DASHBOARD_SCHEMA']}})
factory = async_sessionmaker(engine)
async def session():
    async with factory() as db:
        yield db
app.dependency_overrides[get_db_session] = session
async def serve():
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Measurement.__table__.create)
        now = datetime.now(timezone.utc).replace(microsecond=0)
        async with factory() as db:
            for metric, value in {'bx': 1, 'by': 2, 'bz': -3, 'bt': 5, 'v': 400, 'n': 2, 't': 100000}.items():
                source = 'mag' if metric in ('bx', 'by', 'bz', 'bt') else 'plasma'
                for minutes in (60, 0):
                    db.add(Measurement(metric=metric, value=value, observed_at=now-timedelta(minutes=minutes),
                                       received_at=now, source_product=f'swpc.rtsw_{source}', spacecraft='DSCOVR'))
            for metric, value, hours, source in (
                ('kp', 2, 3, 'swpc.kp'), ('dst', -10, 1, 'swpc.dst'),
                ('ap', 7, 3, 'swpc.kp'), ('f10_7', 145, 24, 'swpc.f107'),
            ):
                db.add(Measurement(metric=metric, value=value, observed_at=now-timedelta(hours=hours),
                                   interval_end=now, received_at=now, source_product=source))
            await db.commit()
        config = uvicorn.Config(app, host='127.0.0.1', port=int(os.environ['TEST_DASHBOARD_PORT']), log_level='error')
        await uvicorn.Server(config).serve()
    finally:
        await engine.dispose()
asyncio.run(serve())
"""
    process = subprocess.Popen([str(root / 'apps/clio/.venv/bin/python'), '-c', code],
        env={**os.environ, 'PYTHONPATH': '', 'TEST_DASHBOARD_SCHEMA': schema,
             'TEST_DASHBOARD_PORT': str(port), 'OBSERVATIONS_SERVICE_TOKEN': 'test-token'})
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('Clio test server exited before readiness')
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=.2):
                    break
            except OSError:
                time.sleep(.1)
        else:
            raise RuntimeError('Clio test server did not become ready')
        yield f'http://127.0.0.1:{port}'
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def get_database_url():
    import os
    from sqlalchemy.engine import make_url
    value = os.getenv('TEST_DATABASE_ADMIN_DSN')
    if not value:
        raise RuntimeError('Set TEST_DATABASE_ADMIN_DSN to an isolated test PostgreSQL server')
    return make_url(value).set(drivername='postgresql+psycopg')


async def verify():
    root = Path(__file__).resolve().parents[4]
    schema = 'dashboard_test_' + uuid4().hex
    admin = create_async_engine(get_database_url())
    engine = create_async_engine(get_database_url(), connect_args={'options': f'-csearch_path={schema}'}, execution_options={'schema_translate_map': {'api': schema, 'clio': schema}})
    factory = async_sessionmaker(engine, expire_on_commit=False)
    created = False
    spec = importlib.util.spec_from_file_location('dashboard_migration', root / 'apps/api/src/api/db/migrations/versions/20260911_0010_dashboard.py')
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    def migrate(connection, direction):
        with Operations.context(MigrationContext.configure(connection)):
            getattr(migration, direction)()
    try:
        async with admin.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA {schema}'))
            created = True
        async with engine.begin() as conn:
            await conn.run_sync(lambda c: migrate(c, 'upgrade'))
        async with factory() as db:
            primary = await create_user(db, UserCreate(username='Admin', password='administrator-password', groups=['admins']))
            other = await create_user(db, UserCreate(username='other', password='other-admin-password', groups=['admins']))
            reader = await create_user(db, UserCreate(username='reader', password='reader-password'))
            now = datetime.now(timezone.utc).replace(microsecond=0)
            await db.commit()
        app = FastAPI()
        app.include_router(router)
        app.include_router(public_router)
        async def session():
            async with factory() as db:
                yield db
        app.dependency_overrides[get_db_session] = session
        transport = ASGITransport(app=app)
        headers = {'Origin': 'https://dashboard.test'}
        with clio_server(root, schema) as clio_url, patch.dict('os.environ', {'DASHBOARD_ORIGINS': 'https://dashboard.test', 'DASHBOARD_COOKIE_SECURE': 'true', 'OBSERVATIONS_URL': clio_url, 'OBSERVATIONS_SERVICE_TOKEN': 'test-token'}):
            async with AsyncClient(transport=transport, base_url='https://dashboard.test', headers=headers) as client, \
                       AsyncClient(transport=transport, base_url='https://dashboard.test', headers=headers) as secondary:
                for params in ({}, {'trends': 'true'}, {'meta': 'true', 'trends': 'true'}):
                    response = await client.get('/observations', params=params)
                    assert response.status_code == 200, response.text
                    data = response.json()['data']
                    assert data['v']['value'] == 400 and data['bt']['value'] == 5
                    assert data['kp']['value'] == 2 and data['f10_7']['value'] == 145
                    assert 'southward_bz' not in data
                    if params.get('trends') == 'true':
                        assert data['changes_1h']['v'] == 0
                for path in ['me', 'users', 'groups', 'api-stats']:
                    assert (await client.get('/dashboard/'+path)).status_code == 401
                async def login(c, username='admin', password='administrator-password'):
                    return await c.post('/dashboard/login', json={'username': username, 'password': password})
                assert (await login(client, password='wrong')).status_code == 401
                response = await login(client)
                assert response.status_code == 200, response.text
                cookie = response.headers['set-cookie']
                assert 'HttpOnly' in cookie and 'Secure' in cookie and 'SameSite=strict' in cookie
                assert response.json()['data']['groups'] == ['admins']
                assert (await client.get('/dashboard/me')).status_code == 200
                assert (await client.post('/dashboard/logout', headers={'Origin': 'https://evil.test'})).status_code == 403
                assert (await client.get('/dashboard/observations')).status_code == 404
                assert (await client.post('/dashboard/users', json={'username': 'admin', 'password': 'some-long-password'})).status_code == 409
                assert (await client.post('/dashboard/users', json={'username': 'new', 'password': 'some-long-password', 'groups': ['unknown']})).status_code == 422
                assert (await client.post('/dashboard/users', json={'username': 'new', 'password': 'some-long-password', 'groups': ['admins']})).status_code == 200
                assert (await client.patch(f"/dashboard/users/{primary['id']}", json={'active': False})).status_code == 409
                assert (await login(secondary, 'reader', 'reader-password')).status_code == 200
                for path in ['users', 'groups', 'api-stats']:
                    assert (await secondary.get('/dashboard/'+path)).status_code == 403
                assert (await client.patch(f"/dashboard/users/{reader['id']}", json={'groups': ['admins']})).status_code == 200
                assert (await secondary.get('/dashboard/me')).status_code == 401
                assert (await login(secondary, 'reader', 'reader-password')).status_code == 200
                assert (await secondary.get('/dashboard/users')).status_code == 200
                assert (await client.patch(f"/dashboard/users/{reader['id']}", json={'active': False})).status_code == 200
                assert (await secondary.get('/dashboard/me')).status_code == 401
                assert (await login(secondary, 'reader', 'reader-password')).status_code == 401
                assert (await login(secondary, 'other', 'other-admin-password')).status_code == 200
                assert (await client.patch(f"/dashboard/users/{other['id']}", json={'password': 'changed-password'})).status_code == 200
                assert (await secondary.get('/dashboard/me')).status_code == 401
                assert (await login(secondary, 'other', 'changed-password')).status_code == 200
                assert (await secondary.post('/dashboard/logout')).status_code == 200
                assert (await secondary.get('/dashboard/me')).status_code == 401
                for i in range(11):
                    response = await login(secondary, 'nonexistent', 'wrong')
                assert response.status_code == 429
                # Visitors share the nginx peer: one visitor must not exhaust a
                # global API peer bucket and block unrelated accounts.
                with patch('api.routers.dashboard.verify_password', return_value=False):
                    for i in range(101):
                        assert (await login(secondary, f'unknown-{i}', 'wrong')).status_code == 401
                    # Changing IPs and username casing must not bypass the account limit.
                    for i in range(11):
                        remote = ASGITransport(app=app, client=(f'198.51.100.{i+1}', 12345))
                        async with AsyncClient(transport=remote, base_url='https://dashboard.test', headers=headers) as distributed:
                            result = await login(distributed, 'Distributed' if i % 2 else 'distributed', 'wrong')
                            assert result.status_code == (401 if i < 10 else 429)
                    # An expired account counter releases the next window.
                    async with factory() as db:
                        await db.execute(update(LoginAttempt).where(LoginAttempt.key == digest('user:distributed'))
                                         .values(window=now-timedelta(minutes=30)))
                        await db.commit()
                    assert (await login(secondary, 'distributed', 'wrong')).status_code == 401
                assert (await login(client)).status_code == 200
                api_statistics.record('/dashboard/users/{user_id}', 'PATCH', 200, 42)
                api_statistics.record('__unmatched__', 'GET', 404, 9)
                with patch.object(api_statistics, 'get_session_factory', return_value=factory):
                    await api_statistics.flush()
                stats = (await client.get('/dashboard/api-stats')).json()['data']
                assert stats['summary']['requests'] == 2 and stats['summary']['errors_4xx'] == 1
                async with factory() as db:
                    await db.execute(update(Session).values(expires_at=now-timedelta(seconds=1)))
                    await db.commit()
                assert (await client.get('/dashboard/me')).status_code == 401
                assert (await client.get('/observations')).status_code == 200
        async with engine.begin() as conn:
            await conn.run_sync(lambda c: migrate(c, 'downgrade'))
            await conn.run_sync(lambda c: migrate(c, 'upgrade'))
        print('Dashboard integration passed: migration roundtrip, login, cookies, CSRF, permissions, users, revocation, pagination, public API, rate limiting and persisted statistics.')
    finally:
        await engine.dispose()
        if created:
            async with admin.begin() as conn:
                await conn.execute(text(f'DROP SCHEMA {schema} CASCADE'))
        await admin.dispose()


if __name__ == '__main__':
    asyncio.run(verify())
