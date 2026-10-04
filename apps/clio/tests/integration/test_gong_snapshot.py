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
from domain_storage import database, migrate, runtime


def configure(environment, monkeypatch, tmp_path):
    for key, value in environment.items():
        if key.startswith('CLIO_DB_'):
            monkeypatch.setenv(key, value)
    monkeypatch.setenv('ARGUS_GONG_ARCHIVE', str(tmp_path / 'gong'))
    monkeypatch.setenv('ARGUS_SDO_ARCHIVE', str(tmp_path / 'sdo'))


def policy():
    return SimpleNamespace(observations={'gong': SimpleNamespace(
        backfill=SimpleNamespace(days=2), sources=SimpleNamespace(live=['gong.live'], historical=['gong.archive']))})


def test_gong_collection_archives_without_features_and_reads_are_causal(database, monkeypatch, tmp_path):
    dsns, urls, environment = database
    migrate(environment)
    configure(environment, monkeypatch, tmp_path)
    from clio.domains import gong
    from clio.providers import gong as provider
    from clio.routers.observation_files import load_observation_files
    from clio.db.session import get_session_factory, dispose_engine
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
            async with get_session_factory()() as session:
                assert not await load_observation_files(session, observed)
                files = await load_observation_files(session, datetime.now(UTC))
                assert len(files) == 1 and files[0].kind == 'gong'
                assert files[0].sha256 == hashlib.sha256(original).hexdigest()
        finally:
            await dispose_engine()
    asyncio.run(check())
    with runtime(dsns, 'clio', urls) as conn:
        row = conn.execute('SELECT raw_path,features,fits_gzip FROM clio.gong_snapshot').fetchone()
        assert (tmp_path / 'gong' / row[0]).read_bytes() == original
        assert row[1:] == (None, None)


def test_migration_preserves_legacy_original_and_first_receipt(database, monkeypatch, tmp_path):
    dsns, urls, environment = database
    configure(environment, monkeypatch, tmp_path)
    command = [sys.executable, '-c', 'from clio.cli import main; main()', 'migrate', 'upgrade']
    subprocess.run([*command, '20261003_unified_measurement'], env=environment, check=True, capture_output=True)
    now = datetime.now(UTC)
    observed, received = now-timedelta(minutes=30), now-timedelta(minutes=20)
    slot = observed.replace(minute=0, second=0, microsecond=0)
    content = b'legacy original'
    digest = hashlib.sha256(content).hexdigest()
    with runtime(dsns, 'clio', urls) as conn:
        conn.execute('''INSERT INTO clio.gong_snapshot
            (slot_at,observed_at,available_at,sha256,source_product,source_url,feature_version,features,fits_gzip)
            VALUES (%s,%s,%s,%s,'gong.live','https://test/legacy.fits.gz','gong-bands-v1','{"field":1}',%s)''',
            (slot, observed, received, digest, gzip.compress(content)))
    subprocess.run([*command, 'head'], env=environment, check=True, capture_output=True)
    from clio.domains import gong
    from clio.providers import gong as provider
    from clio.routers.observation_files import original_file
    from clio.db.session import get_session_factory, dispose_engine
    monkeypatch.setattr(provider, 'candidates', lambda *args, **kwargs: [])
    monkeypatch.setattr(provider, 'download', Mock(side_effect=AssertionError('Legacy original must not be downloaded')))

    async def check():
        try:
            async with get_session_factory()() as session:
                assert (await original_file('gong', digest, session)).body == content
            result = await gong.collect(policy(), mode='live', now=now)
            assert result['files']['gong']['restored'] == 1
        finally:
            await dispose_engine()
    asyncio.run(check())
    with runtime(dsns, 'clio', urls) as conn:
        row = conn.execute('SELECT raw_path,available_at,sha256 FROM clio.gong_snapshot').fetchone()
        assert (tmp_path / 'gong' / row[0]).read_bytes() == content
        assert row[1:] == (received, digest)


def test_goes_source_files_are_persisted_and_served_without_calibration(database, monkeypatch, tmp_path):
    import pandas as pd
    dsns, urls, environment = database
    migrate(environment)
    configure(environment, monkeypatch, tmp_path)
    monkeypatch.setenv('ARGUS_GOES_ARCHIVE', str(tmp_path / 'goes'))
    from clio.domains import goes
    from clio.routers.observation_files import load_observation_files, original_file
    from clio.db.session import get_session_factory, dispose_engine
    now = datetime.now(UTC)
    samples = pd.DataFrame([dict(timestamp=pd.Timestamp(now-timedelta(minutes=10)),
                                goes_euv_256=2.5, goes_euvs_quality_valid=True)])
    source = Mock(return_value=[samples])
    monkeypatch.setattr(goes, 'source_frames', source)
    config = SimpleNamespace(observations={'goes': SimpleNamespace(backfill=SimpleNamespace(days=88))})

    async def check():
        try:
            first = await goes.collect(config, mode='live', now=now)
            assert first['files']['goes']['received'] == 1
            again = await goes.collect(config, mode='live', now=now)
            assert again['files']['goes']['retained'] == 1
            source.assert_called_once()
            async with get_session_factory()() as session:
                refs = await load_observation_files(session, datetime.now(UTC))
                assert len(refs) == 1 and refs[0].kind == 'goes'
                response = await original_file('goes', refs[0].sha256, session)
                assert b'goes_euv_256' in response.body and b's10' not in response.body
        finally:
            await dispose_engine()
    asyncio.run(check())


def test_migration_drops_legacy_aia_table_without_touching_sdo_files(database, tmp_path):
    dsns, urls, environment = database
    command = [sys.executable, '-c', 'from clio.cli import main; main()', 'migrate', 'upgrade']
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
    subprocess.run([*command, 'head'], env=environment, check=True, capture_output=True)
    with runtime(dsns, 'clio', urls) as conn:
        assert conn.execute("SELECT to_regclass('clio.aia_snapshot')").fetchone()[0] is None
        assert conn.execute('SELECT version_num FROM clio.alembic_version').fetchone()[0] == '20261004_drop_aia_snapshot'
        assert conn.execute("SELECT to_regclass('clio.gong_snapshot') IS NOT NULL, to_regclass('clio.goes_snapshot') IS NOT NULL").fetchone() == (True, True)
    assert original.read_bytes() == b'retained SDO original'
    from clio.db.base import Base
    from clio.db import models
    assert 'clio.aia_snapshot' not in Base.metadata.tables
