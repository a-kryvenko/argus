"""Dashboard end-to-end API checks in a disposable PostgreSQL schema.

PYTHONPATH=apps/api apps/api/.venv/bin/python apps/api/tests/integration/verify_dashboard.py
"""
import asyncio
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
from argus_clio.db.models.measurement import Measurement
from argus_clio.db.models.normalized_observation import NormalizedObservation
from app.routers.dashboard import router, create_user, UserCreate
from app.routers.public.observations import router as public_router
from app.services import api_statistics



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
            for model in [Measurement, NormalizedObservation]:
                await conn.run_sync(model.__table__.create)
        async with factory() as db:
            primary = await create_user(db, UserCreate(username='Admin', password='administrator-password', groups=['admins']))
            other = await create_user(db, UserCreate(username='other', password='other-admin-password', groups=['admins']))
            reader = await create_user(db, UserCreate(username='reader', password='reader-password'))
            now = datetime.now(timezone.utc).replace(microsecond=0)
            for metric in ['bx', 'by', 'bz']:
                db.add(Measurement(metric=metric, value=1.0, observed_at=now))
            db.add(NormalizedObservation(observed_at=now, bx=1, by=2, bz=3, v=400, n=2, t=1, kp=1, dst=1, ap=1, f10_7=1))
            await db.commit()
        app = FastAPI()
        app.include_router(router)
        app.include_router(public_router)
        async def session():
            async with factory() as db:
                yield db
        app.dependency_overrides[get_db_session] = session
        from argus_clio.main import app as clio_app
        from argus_clio.db.session import get_db_session as clio_session
        clio_app.dependency_overrides[clio_session] = session
        clio_transport = ASGITransport(app=clio_app)
        transport = ASGITransport(app=app)
        headers = {'Origin': 'https://dashboard.test'}
        with patch.dict('os.environ', {'DASHBOARD_ORIGINS': 'https://dashboard.test', 'DASHBOARD_COOKIE_SECURE': 'true', 'OBSERVATIONS_URL': 'http://clio', 'OBSERVATIONS_SERVICE_TOKEN': 'test-token'}), patch('app.services.observations_client.httpx.AsyncClient', side_effect=lambda **_: AsyncClient(transport=clio_transport)):
            async with AsyncClient(transport=transport, base_url='https://dashboard.test', headers=headers) as client, \
                       AsyncClient(transport=transport, base_url='https://dashboard.test', headers=headers) as secondary:
                assert (await client.get('/public/observations/latest')).status_code == 200
                for path in ['me', 'observations', 'users', 'groups', 'api-stats']:
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
                raw = (await client.get('/dashboard/observations?page_size=2&order=asc')).json()['data']
                assert raw['total'] == 3 and len(raw['items']) == 2
                second = (await client.get('/dashboard/observations?page_size=2&page=2&order=asc')).json()['data']
                assert len(second['items']) == 1 and second['items'][0]['id'] != raw['items'][0]['id']
                assert (await client.get('/dashboard/observations?metric=bz')).json()['data']['total'] == 1
                assert (await client.get('/dashboard/observations?kind=normalized')).json()['data']['total'] == 1
                assert (await client.get('/dashboard/observations?start=2026-01-01')).status_code == 422
                assert (await client.get('/dashboard/observations?page=0')).status_code == 422
                assert (await client.get('/dashboard/observations?start=2099-01-01T00:00:00Z')).json()['data']['total'] == 0
                assert (await client.post('/dashboard/users', json={'username': 'admin', 'password': 'some-long-password'})).status_code == 409
                assert (await client.post('/dashboard/users', json={'username': 'new', 'password': 'some-long-password', 'groups': ['unknown']})).status_code == 422
                assert (await client.post('/dashboard/users', json={'username': 'new', 'password': 'some-long-password', 'groups': ['admins']})).status_code == 200
                assert (await client.patch(f"/dashboard/users/{primary['id']}", json={'active': False})).status_code == 409
                assert (await login(secondary, 'reader', 'reader-password')).status_code == 200
                for path in ['observations', 'users', 'groups', 'api-stats']:
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
