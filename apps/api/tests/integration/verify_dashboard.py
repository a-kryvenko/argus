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
from dotenv import load_dotenv
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.db.session import get_db_session
from app.db.models.dashboard import Session, ApiMetric
from app.routers.dashboard import router, create_user, UserCreate
from app.routers.public.observations import router as public_router
from app.services import api_statistics



@contextmanager
def clio_server(root, schema):
    """Run the observation owner in its own installed environment over real HTTP."""
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    code = """
import os
import uvicorn
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from clio.main import app
from clio.db.session import get_db_session
url = make_url(os.environ['TEST_DATABASE_ADMIN_DSN']).set(drivername='postgresql+psycopg')
engine = create_async_engine(url, execution_options={'schema_translate_map': {'clio': os.environ['TEST_DASHBOARD_SCHEMA']}})
factory = async_sessionmaker(engine)
async def session():
    async with factory() as db:
        yield db
app.dependency_overrides[get_db_session] = session
uvicorn.run(app, host='127.0.0.1', port=int(os.environ['TEST_DASHBOARD_PORT']), log_level='error')
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
    spec = importlib.util.spec_from_file_location('dashboard_migration', root / 'apps/api/alembic/versions/20260911_0010_dashboard.py')
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
            await conn.execute(text('CREATE TABLE normalized_observation (observed_at timestamptz PRIMARY KEY, '
                                    'bx float8, by float8, bz float8, v float8, n float8, t float8, '
                                    'kp float8, dst float8, ap float8, f10_7 float8, s10 float8, m10 float8, y10 float8)'))
        async with factory() as db:
            primary = await create_user(db, UserCreate(username='Admin', password='administrator-password', groups=['admins']))
            other = await create_user(db, UserCreate(username='other', password='other-admin-password', groups=['admins']))
            reader = await create_user(db, UserCreate(username='reader', password='reader-password'))
            now = datetime.now(timezone.utc).replace(microsecond=0)
            await db.execute(text('INSERT INTO normalized_observation (observed_at,bx,by,bz,v,n,t,kp,dst,ap,f10_7) '
                                  'VALUES (:now,1,2,3,400,2,1,1,1,1,1)'), {'now': now})
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
                assert (await client.get('/public/observations/latest')).status_code == 200
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
                assert (await client.get('/public/observations/latest')).status_code == 200
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
