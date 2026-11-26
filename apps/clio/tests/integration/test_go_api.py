from datetime import UTC, datetime, timedelta
import shutil
import sys
from uuid import uuid4

import httpx
import psycopg
from psycopg.types.json import Jsonb
import pytest

from tests.support.clio import clio_server, migrate_clio

LATEST = '/internal/v1/observations/latest'
HISTORY = '/internal/v1/observations/history'
FILES = '/internal/v1/files'
SDO = '/internal/v1/observations/sdo'
EVENTS = '/internal/v1/observations/solar-wind-events/inputs'
TOKEN = 'test-token'
PUBLIC_BASE = 'https://media.example.test/bucket/'


@pytest.fixture
def go_api(database, tmp_path):
    _, urls, environment = database
    if shutil.which('go') is None:
        pytest.skip('Go is required')
    environment.update(OBSERVATIONS_SERVICE_TOKEN=TOKEN, S3_PUBLIC_BASE_URL=PUBLIC_BASE)
    with clio_server(tmp_path, environment) as base_url:
        yield base_url, urls['clio'], environment


def test_go_api_contract(go_api):
    base_url, dsn, environment = go_api
    now = datetime(2026, 11, 24, 12, tzinfo=UTC)
    recent = now + timedelta(minutes=1, microseconds=120000)
    headers = {'Authorization': f'Bearer {TOKEN}'}

    with httpx.Client(base_url=base_url, trust_env=False) as client:
        def check(path, status, **kwargs):
            before = datetime.now(UTC)
            response = client.get(path, **kwargs)
            assert response.status_code == status, response.text
            body = response.json()
            expect_no_store = status == 200 and not path.startswith('/health/')
            assert response.headers.get('cache-control') == ('no-store' if expect_no_store else None)
            if status == 200:
                if path == EVENTS:
                    assert before <= datetime.fromisoformat(body.pop('read_at')) <= datetime.now(UTC)
                elif path.startswith('/health/'):
                    assert body == {'service': 'clio', 'status': 'ok'}
                else:
                    assert body.keys() == {'success', 'data', 'error'}
                    assert body['success'] is True and body['error'] is None
            elif status == 422 and 'detail' in body:
                assert body['detail']
            else:
                assert body['success'] is False
                assert body['error']['code'] == ('NOT_READY' if status == 503 else 'HTTP_ERROR')
                assert body['error']['message']
            return body

        check('/health/live', 200)
        check('/health/ready', 503)
        check(LATEST, 503, headers=headers)
        for path in (HISTORY, FILES, SDO, EVENTS):
            check(path, 401)
        check(SDO, 503, headers=headers)
        check(EVENTS, 503, headers=headers)

        migrate_clio(sys.executable, environment)

        with psycopg.connect(dsn, autocommit=True) as connection:
            check('/health/ready', 200)
            assert (check(LATEST, 200, headers=headers))['data']['points'] == []
            with connection.cursor() as cursor:
                cursor.executemany(
                    'INSERT INTO clio.measurement (metric, observed_at, value, source_product) VALUES (%s, %s, %s, %s)',
                    [
                        ('v', now, 400.5, 'swpc.rtsw_plasma'), ('v', recent, None, 'swpc.rtsw_plasma'),
                        ('v', now + timedelta(hours=1), 999, 'omni.hourly'),
                        ('bz', now, -3.75, 'swpc.rtsw_mag'), ('dst', now, -12.5, None),
                        ('f10_7', now, 109.8, None), ('kp', recent, 10.0, None), ('ap', recent, 2.9, None),
                        ('n', recent, float('inf'), 'swpc.rtsw_plasma'), ('t', recent, float('nan'), 'swpc.rtsw_plasma'),
                        ('s10', recent, 100.0, None), ('s10_81mean', recent, 110.25, None),
                        ('bt', recent, -1.0, 'swpc.rtsw_mag'),
                    ],
                )
            check(LATEST, 401)
            check(LATEST, 401, headers={'Authorization': 'Basic test-token'})
            check(LATEST + '?metrics=invalid', 422, headers=headers)
            check(LATEST + '?metrics=v;bz', 422, headers=headers)
            check(LATEST + '?metrics=invalid&metrics=bz,%20v,bz', 200, headers=headers)
            data = check(LATEST, 200, headers=headers)
            assert data['data']['points'] == [
                {'time': '2026-11-24T12:00:00Z', 'bz': -3.75, 'dst': -12, 'f10_7': 109},
                {'time': '2026-11-24T12:01:00.120000Z', 'v': None, 'kp': None, 'ap': 2,
                 'n': None, 't': None, 's10': None, 's10_81mean': 110.25, 'bt': None},
            ]
            assert type(data['data']['points'][0]['dst']) is int
            check_history(check, connection, now, headers)
            check_explicit_history(check, connection, now, headers)
            check_file_routes(check, connection, now, headers)
            check_events(check, connection, now, headers)


def check_history(check, connection, now, headers):
    start = now - timedelta(minutes=10)
    params = {'from': start.isoformat(), 'to': now.isoformat(), 'metrics': 'bz,v,ap,bt,s10_81mean'}
    check(HISTORY, 422, headers=headers)
    check(HISTORY, 422, headers=headers,
                params={'from': '2026-11-24', 'to': 'abc', 'interval': '10m'})
    check(HISTORY, 422, headers=headers, params={**params, 'from': params['to']})
    assert (check(HISTORY, 200, headers=headers, params=params))['data']['points'] == []
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
    hourly = (check(HISTORY, 200, headers=headers, params=params))['data']
    assert hourly['points'][0] == {
        'time': start.isoformat().replace('+00:00', 'Z'), 'bz': 0., 'v': 400.5,
        'ap': 2, 'bt': 5., 's10_81mean': 105.5,
    }
    check(HISTORY, 200, headers=headers, params={**params, 'interval': '1m'})
    five = (check(HISTORY, 200, headers=headers, params={**params, 'interval': '5m'}))['data']
    assert [(row['v'], row['ap']) for row in five['points']] == [(400, 2), (None, None)]
    check(HISTORY, 200, headers=headers, params={
        **params, 'interval': '5m', 'to': (now - timedelta(minutes=1)).isoformat(),
        'from': (start + timedelta(seconds=1)).isoformat(),
    })
    connection.execute('UPDATE clio.normalized_observation SET bx = 1e200 WHERE observed_at = %s', [start])
    assert (check(HISTORY, 200, headers=headers, params=params))['data']['points'][0]['bt'] is None

    connection.execute('''INSERT INTO clio.measurement (metric, observed_at, value, source_product)
        SELECT 'by', %s::timestamptz + i * interval '1 second', 1, 'swpc.rtsw_mag'
        FROM generate_series(0, 250000) AS i''', [now])
    check(HISTORY, 503, headers=headers, params={
        'metrics': 'by', 'interval': '1m', 'from': now.isoformat(), 'to': (now + timedelta(days=4)).isoformat(),
    })


def check_explicit_history(check, connection, now, headers):
    current_hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    at = min(now - timedelta(days=3), current_hour - timedelta(hours=6))
    at_iso = at.isoformat().replace('+00:00', 'Z')
    with connection.cursor() as cursor:
        cursor.executemany('''INSERT INTO clio.measurement (metric, observed_at, value, source_product)
            VALUES (%s, %s, %s, %s)''', [
            ('v', at, 710, 'omni.hourly'), ('bx', at, 3, 'omni.hourly'),
            ('by', at, -4, 'omni.hourly'), ('bz', at, 12, 'omni.hourly'),
            ('v', at + timedelta(minutes=1), 400, 'swpc.rtsw_plasma'),
            ('v', at + timedelta(minutes=36), 520, 'swpc.rtsw_plasma'),
            ('bt', at + timedelta(minutes=1), 5, 'swpc.rtsw_mag'),
            ('bt', at + timedelta(minutes=36), 7, 'swpc.rtsw_mag'),
            ('v', at + timedelta(hours=1), 777, 'omni.hourly'),
            ('xrs_b', at + timedelta(minutes=3), 2.4e-6, 'goes.xrs'),
            ('proton_10mev', at + timedelta(minutes=5), 0.312, 'goes.protons'),
        ])
    params = {'from': at.isoformat(), 'to': (at + timedelta(hours=2)).isoformat()}
    native = check(HISTORY, 200, headers=headers,
                   params={**params, 'metrics': 'v,bt', 'time_basis': 'l1', 'interval': '1h'})['data']
    assert native == {'time_basis': 'l1', 'interval': '1h', 'points': [{'time': at_iso, 'v': 460, 'bt': 6}]}
    archive = check(HISTORY, 200, headers=headers,
                    params={**params, 'metrics': 'v,bt', 'time_basis': 'near_earth', 'interval': '1h'})['data']
    assert archive == {'time_basis': 'near_earth', 'interval': '1h', 'points': [
        {'time': at_iso, 'v': 710, 'bt': 13},
        {'time': (at + timedelta(hours=1)).isoformat().replace('+00:00', 'Z'), 'v': 777, 'bt': None},
    ]}
    xrs = check(HISTORY, 200, headers=headers,
                params={**params, 'metrics': 'xrs_b', 'interval': '1m'})['data']['points']
    assert [row['xrs_b'] for row in xrs] == [2.4e-6]
    protons = check(HISTORY, 200, headers=headers,
                    params={**params, 'metrics': 'proton_10mev', 'interval': '5m'})['data']['points']
    assert protons == [{'time': (at + timedelta(minutes=5)).isoformat().replace('+00:00', 'Z'), 'proton_10mev': 0.312}]


def add_file_observation(connection, observation, at, key, *, product='test', source_id=None, details=None):
    file_id = uuid4()
    connection.execute('INSERT INTO clio.files (id, timestamp, bucket, key) VALUES (%s, %s, %s, %s)',
                       (file_id, at + timedelta(seconds=1), 'argus-test', key))
    connection.execute('''INSERT INTO clio.file_observation
        (id, observation, observed_at, source_product, source_id, file_id, sha256, details)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)''',
                       (uuid4(), observation, at, product, source_id, file_id, 'a'*64, Jsonb(details)))
    return file_id


def check_file_routes(check, connection, now, headers):
    params = {'observation': 'goes', 'start': now.isoformat(), 'end': (now + timedelta(hours=1)).isoformat()}
    check(FILES, 422, headers=headers)
    check(FILES, 422, headers=headers, params={**params, 'start': params['end']})
    assert (check(FILES, 200, headers=headers, params=params))['data'] == []
    assert (check(SDO, 200, headers=headers))['data'] == {'images': []}
    add_file_observation(connection, 'goes', now, 'goes/z', product='goes.live')
    file_id = add_file_observation(connection, 'goes', now, 'goes/a', product='goes.live',
                                   details={'source_metadata': 'irrelevant'})
    add_file_observation(connection, 'goes', now, 'goes/excluded', source_id='source-event')
    add_file_observation(connection, 'goes', now + timedelta(hours=1), 'goes/excluded-end')
    data = (check(FILES, 200, headers=headers, params=params))['data']
    assert [row['file']['key'] for row in data] == ['goes/a', 'goes/z']
    assert data[0] == {
        'observation': 'goes', 'observed_at': '2026-11-24T12:00:00Z',
        'sha256': 'a'*64, 'source_product': 'goes.live',
        'file': {'id': str(file_id), 'timestamp': '2026-11-24T12:00:01Z',
                 'bucket': 'argus-test', 'key': 'goes/a'},
    }
    assert check(FILES, 200, headers=headers, params={**params, 'observation': 'other'})['data'] == []

    for channel, key in [('193', 'new/é +?@/original.jpg'), ('094', '094/original.jpg')]:
        add_file_observation(connection, 'aia_' + channel + '_browse', now, key, details={
            'channel': channel, 'asset_id': f'2026112412_{channel}_' + 'a'*64,
            'captured_at': now.isoformat(), 'source_updated_at': '2026-11-24T14:00:00+01:00',
        })
    add_file_observation(connection, 'aia_193_browse', now - timedelta(hours=1), 'old/original.jpg')
    images = (check(SDO, 200, headers=headers))['data']['images']
    assert [image['channel'] for image in images] == ['094', '193']
    assert images[1]['urls']['jpg.original'].endswith('/new/%C3%A9%20%2B%3F%40/original.jpg')
    assert 'jpg.4096' not in images[1]['urls']

    add_file_observation(connection, 'aia_193_browse', now + timedelta(hours=1), 'bad/original.jpg',
                         details={'channel': '193', 'asset_id': 'invalid', 'captured_at': now.isoformat()})
    check(SDO, 503, headers=headers)


def check_events(check, connection, now, headers):
    empty = check(EVENTS, 200, headers=headers)
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
    data = check(EVENTS, 200, headers=headers)
    assert data['polls'][0]['poll_id'] == poll_id and data['latest_attempts']['CME']['received_at'] == stamp
    assert data['events'][0]['event_id'] == 'cme1' and data['enlil_arrivals'][0]['simulation_id'] == 'run1'
    connection.execute("UPDATE clio.file_observation SET details = '{\"kind\":\"event\",\"data\":{}}' WHERE source_id = 'cme1'")
    check(EVENTS, 503, headers=headers)
