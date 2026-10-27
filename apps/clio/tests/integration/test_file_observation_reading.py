"""Stage 2 uses only file storage and is ready for legacy table removal."""
import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
import gzip
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import select
from tests.support.database import database, runtime


def migrate(environment, revision='20261027_file_cutover'):
    result = subprocess.run([sys.executable, '-m', 'alembic', '-c',
        str(Path(__file__).resolve().parents[2] / 'alembic.ini'), 'upgrade', revision], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('kind', ['gong', 'goes'])
def test_snapshot_readers_and_collectors_ignore_legacy_and_work_after_its_removal(database, monkeypatch, tmp_path, kind):
    from clio.db.models.file_observation import FileObservation
    from clio.db.session import get_db_session
    from clio.observations.files import store_original, observation_values
    from clio.routers.observation_files import load_observation_files, original_file
    from common.db.session import dispose_engine

    dsns, urls, environment = database
    migrate(environment)
    for key, value in environment.items():
        if key.startswith('CLIO_DB_'):
            monkeypatch.setenv(key, value)
    root = tmp_path / kind
    monkeypatch.setenv(f'ARGUS_{kind.upper()}_ARCHIVE', str(root))
    base = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=3)
    now = base + timedelta(hours=2, minutes=50)
    source = kind + '.live'
    content = b'original observation'
    digest = hashlib.sha256(content).hexdigest()

    def record(hour):
        slot = base + timedelta(hours=hour)
        expected = dict(slot_at=slot, observed_at=slot+timedelta(minutes=10),
            available_at=slot+timedelta(minutes=20), sha256=digest, source_product=source,
            source_url='https://test/observation')
        return store_original(kind, slot, expected['observed_at'], source, lambda: content,
            expected=expected)

    def legacy(values):
        with runtime(dsns, 'clio', urls) as conn:
            if kind == 'gong':
                conn.execute('''INSERT INTO clio.gong_snapshot
                    (slot_at,observed_at,available_at,sha256,source_product,raw_path,source_url)
                    VALUES (%s,%s,%s,%s,%s,%s,'https://test/observation')''', tuple(values[k] for k in (
                        'slot_at', 'observed_at', 'available_at', 'sha256', 'source_product', 'raw_path')))
            else:
                conn.execute('INSERT INTO clio.goes_snapshot VALUES (%s,%s,%s,%s,%s,%s)',
                    tuple(values[k] for k in ('slot_at', 'observed_at', 'available_at', 'sha256', 'source_product', 'raw_path')))

    first = record(0)
    legacy(first)

    async def check():
        try:
            duplicate = record(1)
            legacy(duplicate)
            newest = record(2)
            async with get_db_session() as session:
                session.add_all([FileObservation(**observation_values(kind, r)) for r in (duplicate, newest)])
                await session.commit()
            # A legacy-only receipt with a missing original must not be read.
            (root / first['raw_path']).unlink()
            async with get_db_session() as session:
                refs = await load_observation_files(session, now, kinds=(kind,))
                assert len(refs) == 2 and len({r.slot_at for r in refs}) == 2
                assert len(await load_observation_files(session, base+timedelta(hours=1, minutes=50), kinds=(kind,))) == 1
                assert not await load_observation_files(session, base, kinds=(kind,))
                assert (await original_file(kind, digest, session)).body == content
                # File-only records retain their compressed original.
                archived = await session.scalar(select(FileObservation).where(
                    FileObservation.kind == kind, FileObservation.source_key == observation_values(kind, newest)['source_key']))
                assert (root / archived.raw_path).read_bytes().startswith(b'\x1f\x8b')
            with runtime(dsns, 'clio', urls) as conn:
                assert conn.execute('SELECT count(*) FROM clio.' + kind + '_snapshot').fetchone()[0] == 2
                same_id = conn.execute("SELECT f.id = md5(%s || to_char(legacy.slot_at AT TIME ZONE 'UTC', "
                    "'YYYY-MM-DD\"T\"HH24:MI:SS.US') || ':' || legacy.source_product)::uuid "
                    "FROM clio." + kind + "_snapshot legacy JOIN clio.file_observation f ON f.kind=%s "
                    "WHERE legacy.slot_at=%s AND f.id=%s", (kind + ':', kind, duplicate['slot_at'],
                        observation_values(kind, duplicate)['id'])).fetchone()[0]
                assert same_id  # Stage 1 ON CONFLICT(id) remains safe during rolling writes.
            migrate(environment, 'head')
            if kind == 'gong':
                from clio.domains import gong
                from clio.providers import gong as provider
                monkeypatch.setattr(provider, 'candidates', lambda *a, **kw: [(newest['observed_at'], 'https://test/observation')])
                monkeypatch.setattr(provider, 'download', lambda *a: pytest.fail('Known observation downloaded again'))
                policy = SimpleNamespace(observations={'gong': SimpleNamespace(backfill=SimpleNamespace(days=2),
                    sources=SimpleNamespace(live=[source], historical=['gong.archive']))})
                report = await gong.collect(policy, mode='live', now=now)
            else:
                from clio.domains import goes
                monkeypatch.setattr(goes, 'source_frames', lambda *a: pytest.fail('Known observation downloaded again'))
                policy = SimpleNamespace(observations={'goes': SimpleNamespace(backfill=SimpleNamespace(days=2))})
                report = await goes.collect(policy, mode='live', now=now)
            assert report['files'][kind]['failed'] == 0
            assert report['files'][kind]['received'] == 0
            assert report['files'][kind]['retained'] >= 1
        finally:
            await dispose_engine()
    asyncio.run(check())
    with runtime(dsns, 'clio', urls) as conn:
        assert conn.execute("SELECT to_regclass(%s)", ('clio.' + kind + '_snapshot',)).fetchone()[0] is None


def test_event_inputs_use_only_file_receipts_and_preserve_cutoffs(database, monkeypatch, tmp_path):
    from clio.db.models.file_observation import FileObservation
    from clio.db.session import get_db_session
    from clio.observations.files import store_receipt
    from clio.routers.solar_wind_events import load_event_inputs
    from common.db.session import dispose_engine

    dsns, urls, environment = database
    migrate(environment)
    for key, value in environment.items():
        if key.startswith('CLIO_DB_'):
            monkeypatch.setenv(key, value)
    for kind in ('donki', 'enlil'):
        monkeypatch.setenv(f'ARGUS_{kind.upper()}_ARCHIVE', str(tmp_path / kind))
    base = datetime.now(UTC) - timedelta(hours=3)
    duplicate, latest, failed = uuid4(), uuid4(), uuid4()
    profile_id = uuid4()
    original_poll = b'[{"activityID":"cme1","startTime":"2026-01-01T00:00Z"}]'
    with runtime(dsns, 'clio', urls) as conn:
        conn.execute('''INSERT INTO clio.donki_poll
            (id,endpoint,received_at,request_start,request_end,source_url,sha256,raw_gzip,payload)
            VALUES (%s,'CME',%s,%s,%s,'https://test/CME',%s,%s,%s)''',
            (duplicate, base, base-timedelta(days=30), base, hashlib.sha256(original_poll).hexdigest(),
                gzip.compress(original_poll), original_poll.decode()))

    async def check():
        try:
            async with get_db_session() as session:
                session.add(FileObservation(**store_receipt('donki', 'CME', str(duplicate), id=duplicate,
                    received_at=base, raw=original_poll, metadata=dict(request_start=base-timedelta(days=30), request_end=base))))
                for id, at, error in ((latest, base+timedelta(minutes=20), None), (failed, base+timedelta(minutes=30), 'Timeout')):
                    session.add(FileObservation(**store_receipt('donki', 'CME', str(id), id=id,
                        received_at=at, raw=b'[]' if error is None else None, error=error,
                        metadata=dict(request_start=base-timedelta(days=30), request_end=at))))
                session.add(FileObservation(**store_receipt('enlil', 'noaa.enlil', 'run@' + base.isoformat(), id=profile_id,
                    received_at=base+timedelta(minutes=20), raw=b'netcdf', metadata=dict(
                        source_url='https://test/enlil', source_key='run', source_modified_at=base.isoformat(),
                        model_created_at=base.isoformat(), payload=dict(model_version='2.9', target='L1',
                            density_kind='proton_equivalent', points=[dict(valid_time=(base+timedelta(hours=5)).isoformat(),
                                speed_km_s=500, density_cm3=5)])))))
                await session.commit()
            with runtime(dsns, 'clio', urls) as conn:
                # A later receipt of an older model must not replace the newer model.
                conn.execute('''INSERT INTO clio.enlil_profile_receipt
                    (id,received_at,source_url,source_key,source_modified_at,model_created_at,sha256,raw_gzip,payload)
                    VALUES (%s,%s,'https://test/old-enlil','old-run',%s,%s,%s,%s,%s)''',
                    (uuid4(), base+timedelta(minutes=30), base-timedelta(minutes=10), base-timedelta(hours=1),
                        hashlib.sha256(b'old netcdf').hexdigest(), gzip.compress(b'old netcdf'),
                        json.dumps(dict(model_version='2.8', target='L1', density_kind='proton_equivalent', points=[]))))
                # A late legacy-only response is ignored by the cutover application.
                conn.execute('''INSERT INTO clio.donki_poll
                    (id,endpoint,received_at,request_start,request_end,source_url,sha256,raw_gzip,payload)
                    VALUES (%s,'CME',%s,%s,%s,'https://test/CME',%s,%s,'[]')''',
                    (uuid4(), base+timedelta(minutes=40), base, base, hashlib.sha256(b'[]').hexdigest(), gzip.compress(b'[]')))
            async with get_db_session() as session:
                before = await load_event_inputs(session, base+timedelta(minutes=10), datetime.now(UTC))
                assert len(before.polls) == 1 and before.polls[0].poll_id == duplicate
                assert len(before.polls[0].records) == 1
                assert before.enlil_profiles == []
                current = await load_event_inputs(session, base+timedelta(minutes=35), datetime.now(UTC))
                assert len(current.polls) == 1 and current.polls[0].poll_id == latest
                assert current.polls[0].records == []
                assert current.latest_attempts['CME']['error'] == 'Timeout'
                assert current.enlil_profiles[0].receipt_id == profile_id
                assert current.enlil_profile_status['latest_attempt']['received_at'] == (base+timedelta(minutes=20)).isoformat()
                late = await load_event_inputs(session, base+timedelta(minutes=45), datetime.now(UTC))
                assert late.polls[0].poll_id == latest
                assert late.latest_attempts['CME']['error'] == 'Timeout'
            from clio.domains import enlil as enlil_collection
            monkeypatch.setattr(enlil_collection.enlil, 'discover', lambda now: [dict(
                key='run', size=6, last_modified=base.isoformat())])
            monkeypatch.setattr(enlil_collection.enlil, 'download', lambda url: pytest.fail('Known ENLIL run downloaded again'))
            await enlil_collection.collect()
            # The same application must keep reading and collecting after stage 3.
            migrate(environment, 'head')
            from clio.domains import donki as donki_collection
            policy = SimpleNamespace(lookback_days=30)
            monkeypatch.setattr(donki_collection.donki, 'fetch', lambda *args: (b'[]', []))
            await donki_collection.collect(policy)
            await enlil_collection.store(store_receipt('enlil', 'noaa.enlil', 'discovery', error='Timeout'))
            async with get_db_session() as session:
                inputs = await load_event_inputs(session, datetime.now(UTC), datetime.now(UTC))
                assert len(inputs.polls) == 4
                assert inputs.enlil_profiles[0].receipt_id == profile_id
                assert inputs.enlil_profile_status['latest_attempt']['error'] == 'Timeout'
        finally:
            await dispose_engine()
    asyncio.run(check())
