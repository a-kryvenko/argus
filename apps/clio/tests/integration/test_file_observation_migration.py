"""Move legacy receipts without losing source bytes, identifiers or receipt times."""
from datetime import UTC, datetime, timedelta
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pytest
from tests.support.database import database, runtime


def upgrade(environment, revision):
    return subprocess.run([sys.executable, '-c', 'from clio.cli import main; main()',
        'migrate', 'upgrade', revision], env=environment, capture_output=True, text=True)


def test_migrates_all_four_sources_and_removes_database_payloads(database, tmp_path):
    dsns, urls, environment = database
    roots = {kind: tmp_path / kind for kind in ('goes', 'gong', 'donki', 'enlil')}
    environment.update({f'ARGUS_{kind.upper()}_ARCHIVE': str(root) for kind, root in roots.items()})
    assert upgrade(environment, '20261014_enlil_profile').returncode == 0
    now = datetime.now(UTC)
    raw = {'goes': b'[{"timestamp":"2026-10-27T00:00Z"}]', 'gong': b'original FITS',
           'donki': b'[ {"activityID": "first"} ]', 'enlil': b'original NetCDF'}
    roots['goes'].mkdir()
    (roots['goes'] / 'old.json').write_bytes(raw['goes'])
    poll_id, profile_id, duplicate_id, failed_id = uuid4(), uuid4(), uuid4(), uuid4()
    with runtime(dsns, 'clio', urls) as conn:
        conn.execute('''INSERT INTO clio.goes_snapshot VALUES (%s,%s,%s,%s,%s,%s)''',
            (now, now, now, hashlib.sha256(raw['goes']).hexdigest(), 'goes.live', 'old.json'))
        conn.execute('''INSERT INTO clio.gong_snapshot
            (slot_at,observed_at,available_at,sha256,source_product,source_url,fits_gzip)
            VALUES (%s,%s,%s,%s,'gong.live','https://test/gong',%s)''',
            (now, now, now, hashlib.sha256(raw['gong']).hexdigest(), gzip.compress(raw['gong'])))
        conn.execute('''INSERT INTO clio.donki_poll
            (id,endpoint,received_at,request_start,request_end,source_url,sha256,raw_gzip,payload)
            VALUES (%s,'CME',%s,%s,%s,'https://test/CME',%s,%s,'[]')''',
            (poll_id, now, now-timedelta(days=30), now, hashlib.sha256(raw['donki']).hexdigest(), gzip.compress(raw['donki'])))
        conn.execute('''INSERT INTO clio.donki_poll
            (id,endpoint,received_at,request_start,request_end,source_url,error)
            VALUES (%s,'IPS',%s,%s,%s,'https://test/IPS','Timeout')''', (failed_id, now, now, now))
        conn.execute('''INSERT INTO clio.enlil_profile_receipt
            (id,received_at,source_url,source_key,source_modified_at,model_created_at,sha256,raw_gzip,payload)
            VALUES (%s,%s,'https://test/enlil','run/evo.l1.nc',%s,%s,%s,%s,'{"model_version":"2.9","points":[]}')''',
            (profile_id, now, now, now, hashlib.sha256(raw['enlil']).hexdigest(), gzip.compress(raw['enlil'])))
        conn.execute('''INSERT INTO clio.enlil_profile_receipt
            SELECT %s,received_at,source_url,source_key,source_modified_at,model_created_at,sha256,raw_gzip,payload,error
            FROM clio.enlil_profile_receipt WHERE id=%s''', (duplicate_id, profile_id))
    result = upgrade(environment, 'head')
    assert result.returncode == 0, result.stderr
    with runtime(dsns, 'clio', urls) as conn:
        rows = conn.execute('SELECT id,kind,received_at,raw_path,sha256,error FROM clio.file_observation').fetchall()
        assert len(rows) == 6
        for id, kind, received_at, path, sha256, error in rows:
            assert received_at == now
            if error:
                assert id == failed_id and path is sha256 is None
                continue
            assert gzip.decompress((roots[kind] / path).read_bytes()) == raw[kind]
            assert hashlib.sha256(raw[kind]).hexdigest() == sha256
            receipt = json.loads((roots[kind] / path).with_suffix('.gz.receipt.json').read_text())
            assert receipt['sha256'] == sha256
            if kind == 'donki':
                assert id == poll_id and datetime.fromisoformat(receipt['request_start']) == now-timedelta(days=30)
            if kind == 'enlil':
                assert id in (profile_id, duplicate_id) and receipt['payload']['model_version'] == '2.9'
        columns = {row[0] for row in conn.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='clio' AND table_name='file_observation'")}
        assert columns == {'id', 'kind', 'source_product', 'source_key', 'received_at', 'raw_path', 'sha256', 'error'}
        for table in ('goes_snapshot', 'gong_snapshot', 'donki_poll', 'enlil_profile_receipt'):
            assert conn.execute('SELECT to_regclass(%s)', ('clio.'+table,)).fetchone()[0] is None


def test_corrupt_legacy_original_aborts_migration_without_dropping_receipts(database, tmp_path):
    dsns, urls, environment = database
    environment['ARGUS_GONG_ARCHIVE'] = str(tmp_path)
    assert upgrade(environment, '20261014_enlil_profile').returncode == 0
    now = datetime.now(UTC)
    with runtime(dsns, 'clio', urls) as conn:
        conn.execute('''INSERT INTO clio.gong_snapshot
            (slot_at,observed_at,available_at,sha256,source_product,source_url,fits_gzip)
            VALUES (%s,%s,%s,%s,'gong.live','https://test/gong',%s)''', (now, now, now, '0'*64, gzip.compress(b'wrong bytes')))
    result = upgrade(environment, 'head')
    assert result.returncode != 0 and 'checksum differs' in result.stderr
    with runtime(dsns, 'clio', urls) as conn:
        assert conn.execute('SELECT count(*) FROM clio.gong_snapshot').fetchone()[0] == 1
        assert conn.execute("SELECT to_regclass('clio.file_observation')").fetchone()[0] is None
