"""Run the dashboard application and Go observation path on disposable PostgreSQL."""
import asyncio
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
from shutil import copyfile
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import psycopg
from httpx import ASGITransport, AsyncClient
from sqlalchemy import update

from api.db.models.dashboard import LoginAttempt, Session
from api.db.session import get_db_session
from api.schemas.dashboard import UserCreate
from api.services.authentication import digest
from api.services.dashboard_users import create_user

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tests.support.clio import CLIO_ROOT, clio_server, go_http_server, migrate_clio
from tests.support.database import provisioned_database


def seed_observations(dsn):
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    rows = []
    for metric, value in {'bx': 1, 'by': 2, 'bz': -3, 'bt': 5, 'v': 400, 'n': 2, 't': 100000}.items():
        source = 'mag' if metric in ('bx', 'by', 'bz', 'bt') else 'plasma'
        for minutes in (60, 0):
            rows.append((metric, value, now-timedelta(minutes=minutes), now, f'swpc.rtsw_{source}'))
    for metric, value, hours, source in (
        ('kp', 2, 3, 'swpc.kp'), ('dst', -10, 1, 'swpc.dst'),
        ('ap', 7, 3, 'swpc.kp'), ('f10_7', 145, 24, 'swpc.f107'),
    ):
        observed = now - timedelta(hours=hours)
        if metric == 'dst':
            observed = observed.replace(minute=0)
        elif metric in ('kp', 'ap'):
            observed = observed.replace(hour=observed.hour // 3 * 3, minute=0)
        rows.append((metric, value, observed, now, source))
    with psycopg.connect(dsn) as database, database.cursor() as cursor:
        cursor.executemany("""
            INSERT INTO clio.measurement (metric, value, observed_at, received_at, source_product)
            VALUES (%s, %s, %s, %s, %s)
        """, rows)


@contextmanager
def dashboard_stack():
    with TemporaryDirectory() as directory:
        path = Path(directory)
        with provisioned_database(path) as (_, urls, environment):
            copyfile(ROOT / 'configs/project.yaml', path / 'configs/project.yaml')
            environment.update(
                OBSERVATIONS_SERVICE_TOKEN='test-token', PYTHONPATH='', DEBUG='false',
                DASHBOARD_ORIGINS='https://dashboard.test', DASHBOARD_COOKIE_SECURE='true',
            )
            subprocess.run(
                [sys.executable, '-m', 'alembic', '-c', str(ROOT / 'apps/api/alembic.ini'), 'upgrade', 'head'],
                env=environment, check=True, timeout=60,
            )
            migrate_clio(CLIO_ROOT / '.venv/bin/python', environment)
            seed_observations(urls['clio'])
            with clio_server(path, environment) as clio_url:
                environment['OBSERVATIONS_URL'] = clio_url
                with go_http_server(path, environment, ROOT / 'apps/api/api-go', 'public-api',
                                    'PUBLIC_API_HTTP_ADDR', 'Public API listening on ') as public_url, \
                     patch.dict(os.environ, environment), patch('common.config._config', None):
                    yield public_url


async def verify():
    with dashboard_stack() as public_url:
        from api.main import create_app

        app = create_app()
        async with app.router.lifespan_context(app):
            async with get_db_session() as db:
                primary = await create_user(db, UserCreate(username='Admin', password='administrator-password', groups=['admins']))
                other = await create_user(db, UserCreate(username='other', password='other-admin-password', groups=['admins']))
                reader = await create_user(db, UserCreate(username='reader', password='reader-password'))
                now = datetime.now(UTC).replace(microsecond=0)
                await db.commit()

            transport = ASGITransport(app=app)
            headers = {'Origin': 'https://dashboard.test'}
            async with AsyncClient(transport=transport, base_url='https://dashboard.test', headers=headers) as client, \
                       AsyncClient(transport=transport, base_url='https://dashboard.test', headers=headers) as secondary, \
                       AsyncClient(base_url=public_url, trust_env=False) as public:
                origin = headers['Origin']
                preflight = await client.options('/api/v1/dashboard/login', headers={
                    'Access-Control-Request-Method': 'POST',
                    'Access-Control-Request-Headers': 'content-type',
                })
                assert preflight.status_code == 200, preflight.text
                assert preflight.headers['Access-Control-Allow-Origin'] == origin
                assert preflight.headers['Access-Control-Allow-Credentials'] == 'true'
                assert (await client.get('/api/v1/health/ready')).status_code == 200
                for params in ({}, {'trends': 'true'}):
                    response = await public.get('/api/v1/observations', params=params)
                    assert response.status_code == 200, (params, response.text)
                    data = response.json()['data']
                    assert data['v']['value'] == 400 and data['bt']['value'] == 5
                    assert data['kp']['value'] == 2 and data['f10_7']['value'] == 145
                    assert 'southward_bz' not in data and 'meta' not in data
                    if params.get('trends') == 'true':
                        assert data['changes_1h']['v'] == 0
                for path in ['me', 'users', 'groups']:
                    assert (await client.get('/api/v1/dashboard/'+path)).status_code == 401
                async def login(c, username='admin', password='administrator-password'):
                    return await c.post('/api/v1/dashboard/login', json={'username': username, 'password': password})
                assert (await login(client, password='wrong')).status_code == 401
                response = await login(client)
                assert response.status_code == 200, response.text
                assert response.headers['Cache-Control'] == 'no-store'
                assert response.headers['Access-Control-Allow-Origin'] == origin
                cookie = response.headers['set-cookie']
                assert 'HttpOnly' in cookie and 'Secure' in cookie and 'SameSite=strict' in cookie
                assert response.json()['data']['groups'] == ['admins']
                assert (await client.get('/api/v1/dashboard/me')).status_code == 200
                assert (await client.post('/api/v1/dashboard/logout', headers={'Origin': 'https://evil.test'})).status_code == 403
                assert (await client.get('/api/v1/dashboard/observations')).status_code == 404
                assert (await client.post('/api/v1/dashboard/users', json={'username': 'admin', 'password': 'some-long-password'})).status_code == 409
                assert (await client.post('/api/v1/dashboard/users', json={'username': 'new', 'password': 'some-long-password', 'groups': ['unknown']})).status_code == 422
                assert (await client.post('/api/v1/dashboard/users', json={'username': 'new', 'password': 'some-long-password', 'groups': ['admins']})).status_code == 200
                assert (await client.patch(f"/api/v1/dashboard/users/{primary['id']}", json={'active': False})).status_code == 409
                assert (await login(secondary, 'reader', 'reader-password')).status_code == 200
                for path in ['users', 'groups']:
                    assert (await secondary.get('/api/v1/dashboard/'+path)).status_code == 403
                assert (await client.patch(f"/api/v1/dashboard/users/{reader['id']}", json={'groups': ['admins']})).status_code == 200
                assert (await secondary.get('/api/v1/dashboard/me')).status_code == 401
                assert (await login(secondary, 'reader', 'reader-password')).status_code == 200
                assert (await secondary.get('/api/v1/dashboard/users')).status_code == 200
                assert (await client.patch(f"/api/v1/dashboard/users/{reader['id']}", json={'active': False})).status_code == 200
                assert (await secondary.get('/api/v1/dashboard/me')).status_code == 401
                assert (await login(secondary, 'reader', 'reader-password')).status_code == 401
                assert (await login(secondary, 'other', 'other-admin-password')).status_code == 200
                assert (await client.patch(f"/api/v1/dashboard/users/{other['id']}", json={'password': 'changed-password'})).status_code == 200
                assert (await secondary.get('/api/v1/dashboard/me')).status_code == 401
                assert (await login(secondary, 'other', 'changed-password')).status_code == 200
                assert (await secondary.post('/api/v1/dashboard/logout')).status_code == 200
                assert (await secondary.get('/api/v1/dashboard/me')).status_code == 401
                for i in range(11):
                    response = await login(secondary, 'nonexistent', 'wrong')
                assert response.status_code == 429
                assert int(response.headers['Retry-After']) > 0
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
                    async with get_db_session() as db:
                        await db.execute(update(LoginAttempt).where(LoginAttempt.key == digest('user:distributed'))
                                         .values(window=now-timedelta(minutes=30)))
                        await db.commit()
                    assert (await login(secondary, 'distributed', 'wrong')).status_code == 401
                assert (await login(client)).status_code == 200
                async with get_db_session() as db:
                    await db.execute(update(Session).values(expires_at=now-timedelta(seconds=1)))
                    await db.commit()
                assert (await client.get('/api/v1/dashboard/me')).status_code == 401
                assert (await public.get('/api/v1/observations')).status_code == 200
                assert (await client.get('/api/v1/observations')).status_code == 404

    print('Dashboard and public API integration passed.')


if __name__ == '__main__':
    asyncio.run(verify())
