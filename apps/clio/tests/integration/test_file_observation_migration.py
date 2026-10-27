"""Expand and backfill while old readers and writers remain online."""
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



def export(environment, *args):
    return subprocess.run([sys.executable, '-c', 'from clio.cli import main; main()',
        'backfill-files', *args], env=environment, capture_output=True, text=True)

def test_expansion_preserves_old_readers_and_resumable_backfill_copies_all_sources(database, tmp_path):
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
    status = subprocess.run([sys.executable,
        str(Path(__file__).resolve().parents[4] / 'scripts/deployment/migration_status.py'),
        'clio', '--online'], env=environment, capture_output=True, text=True)
    assert status.returncode == 0, status.stderr
    assert status.stdout.strip() == 'pending'
    result = upgrade(environment, 'head')
    assert result.returncode == 0, result.stderr
    with runtime(dsns, 'clio', urls) as conn:
        assert conn.execute('SELECT count(*) FROM clio.file_observation').fetchone()[0] == 0
        assert conn.execute('SELECT raw_gzip FROM clio.donki_poll WHERE id=%s', (poll_id,)).fetchone()[0]
    assert export(environment, '--check').returncode == 1
    for _ in range(4):
        result = export(environment, '--batch-size', '1')
        assert result.returncode == 0, result.stderr
    result = export(environment, '--check')
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
            assert conn.execute('SELECT to_regclass(%s)', ('clio.'+table,)).fetchone()[0] is not None

    # An old writer can commit an earlier receipt after the first pass.
    late_id = uuid4()
    with runtime(dsns, 'clio', urls) as conn:
        conn.execute('''INSERT INTO clio.donki_poll
            (id,endpoint,received_at,request_start,request_end,source_url,error)
            VALUES (%s,'IPS',%s,%s,%s,'https://test/IPS','Timeout')''', (late_id, now-timedelta(days=1), now, now))
    assert export(environment, '--check').returncode == 1
    assert export(environment).returncode == 0
    assert export(environment, '--check').returncode == 0
    with runtime(dsns, 'clio', urls) as conn:
        assert conn.execute('SELECT count(*) FROM clio.file_observation').fetchone()[0] == 7
        assert conn.execute('SELECT count(*) FROM clio.donki_poll').fetchone()[0] == 3
    # Verification must fail if files are lost after a successful transfer.
    (roots['donki'] / next(r[3] for r in rows if r[1] == 'donki' and r[3])).unlink()
    assert export(environment, '--check').returncode == 1


def test_corrupt_original_aborts_backfill_without_changing_legacy_receipts(database, tmp_path):
    dsns, urls, environment = database
    environment['ARGUS_GONG_ARCHIVE'] = str(tmp_path)
    assert upgrade(environment, '20261014_enlil_profile').returncode == 0
    now = datetime.now(UTC)
    with runtime(dsns, 'clio', urls) as conn:
        conn.execute('''INSERT INTO clio.gong_snapshot
            (slot_at,observed_at,available_at,sha256,source_product,source_url,fits_gzip)
            VALUES (%s,%s,%s,%s,'gong.live','https://test/gong',%s)''', (now, now, now, '0'*64, gzip.compress(b'wrong bytes')))
    result = upgrade(environment, 'head')
    assert result.returncode == 0, result.stderr
    result = export(environment)
    assert result.returncode != 0 and 'checksum differs' in result.stderr
    with runtime(dsns, 'clio', urls) as conn:
        assert conn.execute('SELECT count(*) FROM clio.gong_snapshot').fetchone()[0] == 1
        assert conn.execute('SELECT count(*) FROM clio.file_observation').fetchone()[0] == 0


def test_event_writers_mirror_atomically_and_keep_legacy_readers_working(database, monkeypatch, tmp_path):
    import asyncio
    from types import SimpleNamespace
    from clio.commands import collect_enlil, collect_events
    from clio.db.models.donki_poll import DonkiPoll
    from clio.db.session import get_db_session
    from clio.observations.transition import mirror_observation
    from common.db.session import dispose_engine

    dsns, urls, environment = database
    assert upgrade(environment, 'head').returncode == 0
    for key, value in environment.items():
        if key.startswith('CLIO_DB_'):
            monkeypatch.setenv(key, value)
    for kind in ('donki', 'enlil'):
        monkeypatch.setenv(f'ARGUS_{kind.upper()}_ARCHIVE', str(tmp_path / kind))
    monkeypatch.setattr(collect_events, 'load_observation_config', lambda: SimpleNamespace(
        solar_wind_events=SimpleNamespace(lookback_days=30)))
    monkeypatch.setattr(collect_events.donki, 'fetch', lambda *args: (b'[]', []))
    now = datetime.now(UTC)
    rejected_id = uuid4()

    async def check():
        try:
            await collect_events.run(SimpleNamespace())
            await collect_enlil.store(dict(source_url='https://test/enlil', error='Timeout'))
            async with get_db_session() as session:
                # Mirroring failures roll back the original receipt as well.
                session.add(DonkiPoll(id=rejected_id, endpoint='CME', received_at=now,
                    request_start=now, request_end=now, source_url='https://test/CME',
                    sha256='0'*64, raw_gzip=gzip.compress(b'wrong bytes'), payload=[]))
                await session.flush()
                with pytest.raises(ValueError, match='checksum differs'):
                    await mirror_observation(session, 'donki', id=rejected_id)
                await session.rollback()
        finally:
            await dispose_engine()
    asyncio.run(check())
    with runtime(dsns, 'clio', urls) as conn:
        assert conn.execute('SELECT count(*) FROM clio.donki_poll').fetchone()[0] == 4
        assert conn.execute("SELECT count(*) FROM clio.file_observation WHERE kind='donki'").fetchone()[0] == 4
        assert conn.execute('SELECT count(*) FROM clio.donki_poll WHERE id=%s', (rejected_id,)).fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM clio.file_observation WHERE id=%s', (rejected_id,)).fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM clio.enlil_profile_receipt').fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM clio.file_observation WHERE kind='enlil' AND error='Timeout'").fetchone()[0] == 1
