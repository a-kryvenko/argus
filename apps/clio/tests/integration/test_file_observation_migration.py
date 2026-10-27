"""Cutover is read-only and refuses incomplete or corrupt stage 1 archives."""
import asyncio
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
    return subprocess.run([sys.executable, '-m', 'alembic', '-c',
        str(Path(__file__).resolve().parents[2] / 'alembic.ini'), 'upgrade', revision], env=environment, capture_output=True, text=True)



def seed_legacy(database, monkeypatch, tmp_path):
    dsns, urls, environment = database
    roots = {kind: tmp_path / kind for kind in ('goes', 'gong', 'donki', 'enlil')}
    environment.update({f'ARGUS_{kind.upper()}_ARCHIVE': str(root) for kind, root in roots.items()})
    assert upgrade(environment, '20261027_file_observation').returncode == 0
    for key, value in environment.items():
        if key.startswith('CLIO_DB_') or key.startswith('ARGUS_'):
            monkeypatch.setenv(key, value)
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
    return dsns, urls, environment, roots, raw


def archive_stage1(settings):
    """Prepare the on-disk format and identities produced by the stage 1 release."""
    from psycopg.rows import dict_row
    from clio.db.models.file_observation import FileObservation
    from clio.db.session import get_db_session
    from clio.observations.archive import observation_values, store_original, store_receipt
    from common.db.session import dispose_engine

    dsns, urls, _, _, raw = settings
    values = []
    for kind, table in (('gong', 'gong_snapshot'), ('goes', 'goes_snapshot'),
                        ('donki', 'donki_poll'), ('enlil', 'enlil_profile_receipt')):
        with runtime(dsns, 'clio', urls) as conn:
            rows = conn.cursor(row_factory=dict_row).execute('SELECT * FROM clio.' + table).fetchall()
        for row in rows:
            if kind in ('gong', 'goes'):
                metadata = {k: v for k, v in row.items() if k not in ('fits_gzip', 'raw_path')}
                receipt = store_original(kind, row['slot_at'], row['observed_at'], row['source_product'],
                    lambda: raw[kind], expected=metadata, relative=Path('migrated') / (kind + '.gz'))
                values.append(observation_values(kind, receipt))
            else:
                metadata = {k: v for k, v in row.items() if k not in ('id', 'raw_gzip', 'sha256', 'error')}
                values.append(store_receipt(kind, row['endpoint'] if kind == 'donki' else 'noaa.enlil',
                    str(row['id']), id=row['id'], raw=raw[kind] if row['raw_gzip'] is not None else None,
                    metadata=metadata, received_at=row['received_at'], error=row['error']))

    async def save():
        try:
            async with get_db_session() as session:
                session.add_all([FileObservation(**record) for record in values])
                await session.commit()
        finally:
            await dispose_engine()
    asyncio.run(save())


def test_cutover_refuses_missing_history_then_verifies_without_deleting_legacy_data(database, monkeypatch, tmp_path):
    settings = seed_legacy(database, monkeypatch, tmp_path)
    dsns, urls, environment, roots, _ = settings
    result = upgrade(environment, '20261027_file_cutover')
    assert result.returncode != 0 and 'unarchived or conflicting receipts' in result.stdout + result.stderr
    with runtime(dsns, 'clio', urls) as conn:
        assert conn.execute('SELECT version_num FROM clio.alembic_version').fetchone()[0] == '20261027_file_observation'
    archive_stage1(settings)
    result = upgrade(environment, '20261027_file_cutover')
    assert result.returncode == 0, result.stdout + result.stderr
    assert upgrade(environment, '20261027_file_cutover').returncode == 0
    with runtime(dsns, 'clio', urls) as conn:
        assert conn.execute('SELECT version_num FROM clio.alembic_version').fetchone()[0] == '20261027_file_cutover'
        assert conn.execute('SELECT count(*) FROM clio.file_observation').fetchone()[0] == 6
        assert conn.execute('SELECT count(*) FROM clio.donki_poll').fetchone()[0] == 2
        assert conn.execute('SELECT count(*) FROM clio.enlil_profile_receipt').fetchone()[0] == 2
        assert conn.execute('SELECT fits_gzip FROM clio.gong_snapshot').fetchone()[0]
        assert conn.execute('SELECT raw_path FROM clio.goes_snapshot').fetchone()[0] == 'old.json'
    assert (roots['goes'] / 'old.json').exists()
    with runtime(dsns, 'clio', urls) as conn:
        before = conn.execute('SELECT id,kind,raw_path,sha256 FROM clio.file_observation ORDER BY id').fetchall()
    originals = {(kind, path): (roots[kind] / path).read_bytes() for _, kind, path, _ in before if path}
    result = upgrade(environment, '20261027_drop_legacy_files')
    assert result.returncode == 0, result.stdout + result.stderr
    with runtime(dsns, 'clio', urls) as conn:
        for table in ('gong_snapshot', 'goes_snapshot', 'donki_poll', 'enlil_profile_receipt'):
            assert conn.execute('SELECT to_regclass(%s)', ('clio.' + table,)).fetchone()[0] is None
        assert conn.execute('SELECT id,kind,raw_path,sha256 FROM clio.file_observation ORDER BY id').fetchall() == before
    assert all((roots[kind] / path).read_bytes() == content for (kind, path), content in originals.items())


@pytest.mark.parametrize('damage', ['gzip', 'missing', 'sidecar', 'time'])
def test_cutover_refuses_corrupt_archive_before_advancing_revision(database, monkeypatch, tmp_path, damage):
    settings = seed_legacy(database, monkeypatch, tmp_path)
    dsns, urls, environment, roots, _ = settings
    archive_stage1(settings)
    path = roots['gong'] / 'migrated/gong.gz'
    if damage == 'gzip':
        path.write_bytes(gzip.compress(b'wrong original'))
    elif damage == 'missing':
        path.unlink()
    elif damage == 'sidecar':
        path.with_suffix('.gz.receipt.json').unlink()
    else:
        receipt = path.with_suffix('.gz.receipt.json')
        metadata = json.loads(receipt.read_text())
        metadata['available_at'] = datetime.now(UTC).isoformat()
        receipt.write_text(json.dumps(metadata))
    result = upgrade(environment, '20261027_file_cutover')
    assert result.returncode != 0 and 'cutover blocked' in result.stdout + result.stderr
    with runtime(dsns, 'clio', urls) as conn:
        assert conn.execute('SELECT version_num FROM clio.alembic_version').fetchone()[0] == '20261027_file_observation'
        assert conn.execute('SELECT count(*) FROM clio.gong_snapshot').fetchone()[0] == 1
