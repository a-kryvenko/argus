"""GONG originals and receipts survive the transition to file-only collection."""
import asyncio
from datetime import UTC, datetime, timedelta
import gzip
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.support.database import database, migrate, runtime


def configure(environment, monkeypatch, tmp_path):
    for key, value in environment.items():
        if key.startswith('CLIO_DB_'):
            monkeypatch.setenv(key, value)
    monkeypatch.setenv('ARGUS_GONG_ARCHIVE', str(tmp_path / 'gong'))
    monkeypatch.setenv('ARGUS_SDO_ARCHIVE', str(tmp_path / 'sdo'))


def policy():
    return SimpleNamespace(observations={'gong': SimpleNamespace(
        backfill_days=2, sources=SimpleNamespace(live=['gong.live'], historical=['gong.archive']))})


def test_gong_collection_archives_without_features_and_reads_are_causal(database, monkeypatch, tmp_path):
    dsns, urls, environment = database
    migrate(environment)
    configure(environment, monkeypatch, tmp_path)
    from clio.ingestion import gong
    from clio.providers import gong as provider
    from clio.observations.file_catalog import load_observation_files
    from clio.db.session import get_db_session
    from common.db.session import dispose_engine
    now = datetime.now(UTC)
    observed = now-timedelta(minutes=30)
    original = b'original FITS for Prophet'
    download = Mock(return_value=original)
    monkeypatch.setattr(provider, 'download', download)
    monkeypatch.setattr(provider, 'candidates', lambda *args, **kwargs: [(observed, 'https://test/file.fits.gz')])

    async def check():
        try:
            first = await gong.collect(policy(), mode='live', now=now)
            assert first['files']['gong']['received'] == 1
            again = await gong.collect(policy(), mode='backfill', now=now)
            assert again['files']['gong']['retained'] == 1
            download.assert_called_once()
            async with get_db_session() as session:
                assert not await load_observation_files(session, observed)
                files = await load_observation_files(session, datetime.now(UTC))
                assert len(files) == 1 and files[0].kind == 'gong'
                assert files[0].sha256 == hashlib.sha256(original).hexdigest()
        finally:
            await dispose_engine()
    asyncio.run(check())
    with runtime(dsns, 'clio', urls) as conn:
        assert conn.execute("SELECT to_regclass('clio.gong_snapshot')").fetchone()[0] is None
        archived = conn.execute("SELECT raw_path,sha256,received_at FROM clio.file_observation WHERE kind='gong'").fetchone()
        assert gzip.decompress((tmp_path / 'gong' / archived[0]).read_bytes()) == original
        assert archived[1] == hashlib.sha256(original).hexdigest()



def test_goes_source_files_are_persisted_and_served_without_calibration(database, monkeypatch, tmp_path):
    import pandas as pd
    dsns, urls, environment = database
    migrate(environment)
    configure(environment, monkeypatch, tmp_path)
    monkeypatch.setenv('ARGUS_GOES_ARCHIVE', str(tmp_path / 'goes'))
    from clio.ingestion import goes
    from clio.observations.file_catalog import load_observation_files
    from clio.routers.observation_files import original_file
    from clio.db.session import get_db_session
    from common.db.session import dispose_engine
    now = datetime.now(UTC)
    samples = pd.DataFrame([dict(timestamp=pd.Timestamp(now-timedelta(minutes=10)),
                                goes_euv_256=2.5, goes_euvs_quality_valid=True)])
    source = Mock(return_value=[samples])
    monkeypatch.setattr(goes, 'source_frames', source)
    config = SimpleNamespace(observations={'goes': SimpleNamespace(backfill_days=88)})

    async def check():
        try:
            first = await goes.collect(config, mode='live', now=now)
            assert first['files']['goes']['received'] == 1
            again = await goes.collect(config, mode='live', now=now)
            assert again['files']['goes']['retained'] == 1
            source.assert_called_once()
            async with get_db_session() as session:
                refs = await load_observation_files(session, datetime.now(UTC))
                assert len(refs) == 1 and refs[0].kind == 'goes'
                response = await original_file('goes', refs[0].sha256, session)
                assert b'goes_euv_256' in response.body and b's10' not in response.body
        finally:
            await dispose_engine()
    asyncio.run(check())


def test_migration_drops_legacy_aia_table_without_touching_sdo_files(database, tmp_path):
    dsns, urls, environment = database
    command = [sys.executable, '-m', 'alembic', '-c',
               str(Path(__file__).resolve().parents[2] / 'alembic.ini'), 'upgrade']
    subprocess.run([*command, '20261004_raw_observation_files'], env=environment,
                   check=True, capture_output=True)
    now = datetime.now(UTC)
    original = tmp_path / 'sdo' / 'aia193.fits'
    original.parent.mkdir()
    original.write_bytes(b'retained SDO original')
    with runtime(dsns, 'clio', urls) as conn:
        conn.execute('''INSERT INTO clio.aia_snapshot
            (slot_at,observed_at,available_at,sha256,raw_path)
            VALUES (%s,%s,%s,%s,%s)''', (now, now, now, 'a'*64, str(original)))
    subprocess.run([*command, '20261004_drop_aia_snapshot'], env=environment, check=True, capture_output=True)
    with runtime(dsns, 'clio', urls) as conn:
        assert conn.execute("SELECT to_regclass('clio.aia_snapshot')").fetchone()[0] is None
        assert conn.execute("SELECT to_regclass('clio.gong_snapshot') IS NOT NULL, to_regclass('clio.goes_snapshot') IS NOT NULL").fetchone() == (True, True)
    assert original.read_bytes() == b'retained SDO original'
    from clio.db.base import Base
    assert 'aia_snapshot' not in Base.metadata.tables
