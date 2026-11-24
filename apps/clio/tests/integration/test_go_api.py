import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
from uuid import uuid4

import httpx
import psycopg
from psycopg.types.json import Jsonb
import pytest

GO_API = Path(__file__).resolve().parents[2] / 'api-go'
LATEST = '/internal/v1/observations/latest'
HISTORY = '/internal/v1/observations/history'
FILES = '/internal/v1/files'
SDO = '/internal/v1/observations/sdo'
EVENTS = '/internal/v1/observations/solar-wind-events/inputs'
TOKEN = 'parity-token'
PUBLIC_BASE = 'https://media.example.test/bucket/'


@pytest.fixture
def go_api(database, monkeypatch, tmp_path):
    _, urls, environment = database
    go = shutil.which('go')
    if go is None:
        pytest.skip('Go is required')

    binary = tmp_path / 'clio-api'
    subprocess.run([go, 'build', '-o', str(binary), '.'], cwd=GO_API, check=True, timeout=120)
    environment.update(OBSERVATIONS_SERVICE_TOKEN=TOKEN, S3_PUBLIC_BASE_URL=PUBLIC_BASE)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    with subprocess.Popen([binary], env={**environment, 'CLIO_HTTP_ADDR': '127.0.0.1:0'},
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                assert selector.select(timeout=10), 'Go API did not start'
                line = process.stdout.readline().rstrip()
            _, marker, address = line.partition('Clio listening on ')
            assert marker, line
            yield f'http://{address}', urls['clio']
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def test_go_api_matches_fastapi(go_api):
    from clio.main import app
    from clio.db.session import dispose_sync_engine
    from common.db.session import dispose_engine

    base_url, dsn = go_api
    now = datetime(2026, 11, 24, 12, tzinfo=UTC)
    recent = now + timedelta(minutes=1, microseconds=120000)
    headers = {'Authorization': f'Bearer {TOKEN}'}

    async def compare():
        async with (
            httpx.AsyncClient(base_url=base_url, trust_env=False) as go,
            httpx.AsyncClient(base_url='http://clio', transport=httpx.ASGITransport(app=app)) as python,
        ):
            async def check(path, status, **kwargs):
                before = datetime.now(UTC)
                expected = await python.get(path, **kwargs)
                actual = await go.get(path, **kwargs)
                assert actual.status_code == expected.status_code == status
                assert actual.headers.get('cache-control') == expected.headers.get('cache-control')
                expected_body, actual_body = expected.json(), actual.json()
                if path == EVENTS and status == 200:
                    after = datetime.now(UTC)
                    for body in (expected_body, actual_body):
                        assert before <= datetime.fromisoformat(body.pop('read_at')) <= after
                assert actual_body == expected_body
                return actual_body

            await check('/health/live', 200)
            await check('/health/ready', 503)
            await check(LATEST, 503, headers=headers)
            for path in (HISTORY, FILES, SDO, EVENTS):
                await check(path, 401)
            await check(SDO, 503, headers=headers)
            await check(EVENTS, 503, headers=headers)

            subprocess.run([
                sys.executable, '-m', 'alembic', '-c', str(GO_API.parent / 'alembic.ini'), 'upgrade', 'head',
            ], check=True, timeout=60)

            with psycopg.connect(dsn, autocommit=True) as connection:
                await check('/health/ready', 200)
                assert (await check(LATEST, 200, headers=headers))['data']['points'] == []
                with connection.cursor() as cursor:
                    cursor.executemany(
                        'INSERT INTO clio.measurement (metric, observed_at, value) VALUES (%s, %s, %s)',
                        [
                            ('v', now, 400.5), ('v', recent, None), ('bz', now, -3.75),
                            ('dst', now, -12.5), ('f10_7', now, 109.8), ('kp', recent, 10.0),
                            ('ap', recent, 2.9), ('n', recent, float('inf')), ('t', recent, float('nan')),
                            ('s10', recent, 100.0), ('s10_81mean', recent, 110.25), ('bt', recent, -1.0),
                        ],
                    )
                await check(LATEST, 401)
                await check(LATEST, 401, headers={'Authorization': 'Basic parity-token'})
                await check(LATEST + '?metrics=invalid', 422, headers=headers)
                await check(LATEST + '?metrics=v;bz', 422, headers=headers)
                await check(LATEST + '?metrics=invalid&metrics=bz,%20v,bz', 200, headers=headers)
                data = await check(LATEST, 200, headers=headers)
                assert data['data']['points'] == [
                    {'time': '2026-11-24T12:00:00Z', 'bz': -3.75, 'dst': -12, 'f10_7': 109},
                    {'time': '2026-11-24T12:01:00.120000Z', 'v': None, 'kp': None, 'ap': 2,
                     'n': None, 't': None, 's10': None, 's10_81mean': 110.25, 'bt': None},
                ]
                assert type(data['data']['points'][0]['dst']) is int
                await check_history(check, connection, now, headers)
                await check_file_routes(check, connection, now, headers)
                await check_events(check, connection, now, headers)

    async def run():
        try:
            await compare()
        finally:
            await dispose_engine()
            dispose_sync_engine()

    asyncio.run(run())


async def check_history(check, connection, now, headers):
    start = now - timedelta(minutes=10)
    params = {'from': start.isoformat(), 'to': now.isoformat(), 'metrics': 'bz,v,ap,bt,s10_81mean'}
    await check(HISTORY, 422, headers=headers)
    await check(HISTORY, 422, headers=headers,
                params={'from': '2026-11-24', 'to': 'abc', 'interval': '10m'})
    await check(HISTORY, 422, headers=headers, params={**params, 'from': params['to']})
    assert (await check(HISTORY, 200, headers=headers, params=params))['data']['points'] == []
    connection.execute('''INSERT INTO clio.normalized_observation
        (observed_at, bx, by, bz, v, n, t, kp, dst, ap, f10_7)
        VALUES (%s, -3.0, 4.0, 0.0, 400.5, 1.0, 3000.0, 1.9, -12.5, 2.9, 110.0)
    ''', [start])
    with connection.cursor() as cursor:
        cursor.executemany('''INSERT INTO clio.measurement (metric, observed_at, value, source_product)
            VALUES (%s, %s, %s, %s)''', [
            ('v', start, 200, 'swpc.rtsw_plasma'),
            ('v', start + timedelta(seconds=30), 300, 'swpc.rtsw_plasma'),
            ('v', start + timedelta(minutes=1), 500, 'swpc.rtsw_plasma'),
            ('v', start + timedelta(minutes=2), 999, 'omni.hourly'),
            ('v', start + timedelta(minutes=5), float('inf'), 'swpc.rtsw_plasma'),
            ('ap', start, 1.9, 'swpc.kp'),
            ('ap', start + timedelta(minutes=1), 2.9, 'swpc.kp'),
            ('s10_81mean', start, 105.5, 'solfsmy.means_archive'),
            ('s10_81mean', start + timedelta(minutes=7), 110.5, 'solfsmy.means'),
        ])
    hourly = (await check(HISTORY, 200, headers=headers, params=params))['data']
    assert hourly['points'][0] == {
        'time': start.isoformat().replace('+00:00', 'Z'), 'bz': 0., 'v': 400.5,
        'ap': 2, 'bt': 5., 's10_81mean': 105.5,
    }
    await check(HISTORY, 200, headers=headers, params={**params, 'interval': '1m'})
    five = (await check(HISTORY, 200, headers=headers, params={**params, 'interval': '5m'}))['data']
    assert [(row['v'], row['ap']) for row in five['points']] == [(400, 2), (None, None)]
    await check(HISTORY, 200, headers=headers, params={
        **params, 'interval': '5m', 'to': (now - timedelta(minutes=1)).isoformat(),
        'from': (start + timedelta(seconds=1)).isoformat(),
    })
    connection.execute('UPDATE clio.normalized_observation SET bx = 1e200 WHERE observed_at = %s', [start])
    assert (await check(HISTORY, 200, headers=headers, params=params))['data']['points'][0]['bt'] is None

    connection.execute('''INSERT INTO clio.measurement (metric, observed_at, value, source_product)
        SELECT 'by', %s::timestamptz + i * interval '1 second', 1, 'swpc.rtsw_mag'
        FROM generate_series(0, 250000) AS i''', [now])
    await check(HISTORY, 503, headers=headers, params={
        'metrics': 'by', 'interval': '1m', 'from': now.isoformat(), 'to': (now + timedelta(days=4)).isoformat(),
    })


def add_file_observation(connection, observation, at, key, *, product='test', source_id=None, details=None):
    file_id = uuid4()
    connection.execute('INSERT INTO clio.files (id, timestamp, bucket, key) VALUES (%s, %s, %s, %s)',
                       (file_id, at + timedelta(seconds=1), 'argus-test', key))
    connection.execute('''INSERT INTO clio.file_observation
        (id, observation, observed_at, source_product, source_id, file_id, sha256, details)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)''',
                       (uuid4(), observation, at, product, source_id, file_id, 'a'*64, Jsonb(details)))


async def check_file_routes(check, connection, now, headers):
    params = {'observation': 'goes', 'start': now.isoformat(), 'end': (now + timedelta(hours=1)).isoformat()}
    await check(FILES, 422, headers=headers)
    await check(FILES, 422, headers=headers, params={**params, 'start': params['end']})
    assert (await check(FILES, 200, headers=headers, params=params))['data'] == []
    assert (await check(SDO, 200, headers=headers))['data'] == {'images': []}
    add_file_observation(connection, 'goes', now, 'goes/z', product='goes.live')
    add_file_observation(connection, 'goes', now, 'goes/a', product='goes.live')
    add_file_observation(connection, 'goes', now, 'goes/excluded', source_id='source-event')
    add_file_observation(connection, 'goes', now + timedelta(hours=1), 'goes/excluded-end')
    data = (await check(FILES, 200, headers=headers, params=params))['data']
    assert [row['file']['key'] for row in data] == ['goes/a', 'goes/z']

    for channel, key in [('193', 'new/é +?@/original.jpg'), ('094', '094/original.jpg')]:
        add_file_observation(connection, 'aia_' + channel + '_browse', now, key, details={
            'channel': channel, 'asset_id': f'2026112412_{channel}_' + 'a'*64,
            'captured_at': now.isoformat(), 'source_updated_at': '2026-11-24T14:00:00+01:00',
        })
    add_file_observation(connection, 'aia_193_browse', now - timedelta(hours=1), 'old/original.jpg')
    images = (await check(SDO, 200, headers=headers))['data']['images']
    assert [image['channel'] for image in images] == ['094', '193']
    assert images[1]['urls']['jpg.original'].endswith('/new/%C3%A9%20%2B%3F%40/original.jpg')
    assert 'jpg.4096' not in images[1]['urls']

    add_file_observation(connection, 'aia_193_browse', now + timedelta(hours=1), 'bad/original.jpg',
                         details={'channel': '193', 'asset_id': 'invalid', 'captured_at': now.isoformat()})
    await check(SDO, 503, headers=headers)


async def check_events(check, connection, now, headers):
    empty = await check(EVENTS, 200, headers=headers)
    assert empty == {'schema_version': 2, 'polls': [], 'events': [], 'enlil_arrivals': [], 'latest_attempts': {}}
    poll_id = str(uuid4())
    stamp = now.isoformat()
    receipt = {'poll_id': poll_id, 'sha256': 'a'*64, 'received_at': stamp}
    add_file_observation(connection, 'donki.CME', now, 'donki/poll', product='CME', source_id='', details={
        'poll_id': poll_id, 'request_start': (now - timedelta(days=30)).isoformat(),
        'request_end': stamp, 'record_count': 1,
    })
    add_file_observation(connection, 'donki.CME', now, 'donki/cme1', product='CME', source_id='cme1', details={
        'kind': 'event', 'data': {'event_id': 'cme1', 'kind': 'cme', 'status': 'detected', 'source': 'DONKI',
                                  'available_at': stamp, 'arrival_receipts': [receipt]},
    })
    add_file_observation(connection, 'donki.WSAEnlilSimulations', now, 'donki/run1', product='WSAEnlilSimulations',
                         source_id='run1', details={'kind': 'arrival', 'data': {
                             'simulation_id': 'run1', 'available_at': stamp, 'model_completed_at': stamp,
                             'cme_ids': ['cme1'], 'receipts': [receipt],
                         }})
    data = await check(EVENTS, 200, headers=headers)
    assert data['polls'][0]['poll_id'] == poll_id and data['latest_attempts']['CME']['received_at'] == stamp
    assert data['events'][0]['event_id'] == 'cme1' and data['enlil_arrivals'][0]['simulation_id'] == 'run1'
    connection.execute("UPDATE clio.file_observation SET details = '{\"kind\":\"event\",\"data\":{}}' WHERE source_id = 'cme1'")
    await check(EVENTS, 503, headers=headers)
