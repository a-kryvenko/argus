"""Dashboard end-to-end API checks in a disposable PostgreSQL schema.

apps/api/.venv/bin/python apps/api/tests/integration/verify_dashboard.py
"""
import asyncio
from contextlib import contextmanager
import os
from shutil import copyfile
import sys
from tempfile import TemporaryDirectory
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

import psycopg
from alembic.migration import MigrationContext
from alembic.operations import Operations
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from api.db.session import get_request_db_session
from api.db.models.dashboard import Session, LoginAttempt
from api.services.authentication import digest
from api.routers.dashboard import router
from api.schemas.dashboard import UserCreate
from api.services.authentication import create_user
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from apps.api.tests.reference_public.routers.observations import router as public_router
from tests.support.clio import CLIO_ROOT, clio_server, go_http_server, migrate_clio
from tests.support.database import provisioned_database


@contextmanager
def dashboard_clio():
    with TemporaryDirectory() as directory:
        path = Path(directory)
        with provisioned_database(path) as (_, urls, environment):
            copyfile(ROOT / 'configs/project.yaml', path / 'configs/project.yaml')
            environment.update(OBSERVATIONS_SERVICE_TOKEN='test-token', PYTHONPATH='')
            migrate_clio(CLIO_ROOT / '.venv/bin/python', environment)
            now = datetime.now(timezone.utc).replace(microsecond=0)
            rows = []
            for metric, value in {'bx': 1, 'by': 2, 'bz': -3, 'bt': 5, 'v': 400, 'n': 2, 't': 100000}.items():
                source = 'mag' if metric in ('bx', 'by', 'bz', 'bt') else 'plasma'
                for minutes in (60, 0):
                    rows.append((metric, value, now-timedelta(minutes=minutes), now, f'swpc.rtsw_{source}'))
            for metric, value, hours, source in (
                ('kp', 2, 3, 'swpc.kp'), ('dst', -10, 1, 'swpc.dst'),
                ('ap', 7, 3, 'swpc.kp'), ('f10_7', 145, 24, 'swpc.f107'),
            ):
                rows.append((metric, value, now-timedelta(hours=hours), now, source))
            with psycopg.connect(urls['clio']) as database, database.cursor() as cursor:
                cursor.executemany("""
                    INSERT INTO clio.measurement (metric, value, observed_at, received_at, source_product)
                    VALUES (%s, %s, %s, %s, %s)
                """, rows)
            with clio_server(path, environment) as url:
                environment['OBSERVATIONS_URL'] = url
                with go_http_server(path, environment, ROOT / 'apps/api/api-go', 'public-api', 'PUBLIC_API_HTTP_ADDR',
                                    'Public API listening on ') as public_url, \
                     patch.dict(os.environ, {'ARGUS_WORKDIR': str(path)}), patch('common.config._config', None):
                    yield url, public_url


def get_database_url():
    from sqlalchemy.engine import make_url
    value = os.getenv('TEST_DATABASE_ADMIN_DSN')
    if not value:
        raise RuntimeError('Set TEST_DATABASE_ADMIN_DSN to an isolated test PostgreSQL server')
    return make_url(value).set(drivername='postgresql+psycopg')


async def verify():
    schema = 'dashboard_test_' + uuid4().hex
    admin = create_async_engine(get_database_url())
    engine = create_async_engine(get_database_url(), connect_args={'options': f'-csearch_path={schema}'})
    factory = async_sessionmaker(engine, expire_on_commit=False)
    created = False
    spec = importlib.util.spec_from_file_location('dashboard_migration', ROOT / 'apps/api/src/api/db/migrations/versions/20260911_0010_dashboard.py')
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
        reference_api = FastAPI()
        reference_api.include_router(public_router)
        async def session():
            async with factory() as db:
                yield db
        app.dependency_overrides[get_request_db_session] = session
        transport = ASGITransport(app=app)
        headers = {'Origin': 'https://dashboard.test'}
        with dashboard_clio() as (clio_url, public_url), patch.dict('os.environ', {'DASHBOARD_ORIGINS': 'https://dashboard.test', 'DASHBOARD_COOKIE_SECURE': 'true', 'OBSERVATIONS_URL': clio_url, 'OBSERVATIONS_SERVICE_TOKEN': 'test-token'}):
            async with AsyncClient(transport=transport, base_url='https://dashboard.test', headers=headers) as client, \
                       AsyncClient(transport=transport, base_url='https://dashboard.test', headers=headers) as secondary, \
                       AsyncClient(base_url=public_url, trust_env=False) as public, \
                       AsyncClient(transport=ASGITransport(app=reference_api), base_url='http://reference') as reference:
                for params in ({}, {'trends': 'true'}, {'meta': 'true', 'trends': 'true'}):
                    response = await public.get('/api/v1/observations', params=params)
                    assert response.status_code == 200, (params, response.text)
                    data = response.json()['data']
                    expected = (await reference.get('/observations', params=params)).json()['data']
                    assert data == expected, (params, data, expected)
                    assert data['v']['value'] == 400 and data['bt']['value'] == 5
                    assert data['kp']['value'] == 2 and data['f10_7']['value'] == 145
                    assert 'southward_bz' not in data
                    if params.get('trends') == 'true':
                        assert data['changes_1h']['v'] == 0
                for path in ['me', 'users', 'groups']:
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
                for path in ['users', 'groups']:
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
                async with factory() as db:
                    await db.execute(update(Session).values(expires_at=now-timedelta(seconds=1)))
                    await db.commit()
                assert (await client.get('/dashboard/me')).status_code == 401
                assert (await public.get('/api/v1/observations')).status_code == 200
                assert (await client.get('/observations')).status_code == 404
        async with engine.begin() as conn:
            await conn.run_sync(lambda c: migrate(c, 'downgrade'))
            await conn.run_sync(lambda c: migrate(c, 'upgrade'))
        print('Dashboard integration passed: migration roundtrip, login, cookies, CSRF, permissions, users, revocation, pagination, public API, rate limiting.')
    finally:
        await engine.dispose()
        if created:
            async with admin.begin() as conn:
                await conn.execute(text(f'DROP SCHEMA {schema} CASCADE'))
        await admin.dispose()


if __name__ == '__main__':
    asyncio.run(verify())
