"""GONG originals, receipts and reads use Clio's isolated database."""
import asyncio
from datetime import UTC, datetime, timedelta
import gzip
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from domain_storage import database, migrate, runtime


def test_gong_collection_is_immutable_and_reads_are_causal(database, monkeypatch):
    dsns, urls, environment = database
    migrate(environment)
    for key, value in environment.items():
        if key.startswith('CLIO_DB_'):
            monkeypatch.setenv(key, value)
    from clio.domains import gong
    from clio.providers import gong as provider
    from clio.db.session import get_session_factory, dispose_engine
    now = datetime(2026, 9, 29, 12, tzinfo=UTC)
    observed, received = now-timedelta(minutes=30), now-timedelta(minutes=20)
    values = dict(slot_at=observed.replace(minute=0), observed_at=observed, available_at=received,
                  sha256='a'*64, source_product='gong.live', source_url='https://test/file.fits.gz',
                  feature_version='gong-bands-v1', features={'field': 1.}, fits_gzip=gzip.compress(b'original'))
    monkeypatch.setattr(provider, 'candidates', lambda *args, **kwargs: [(observed, values['source_url'])])
    snapshot = Mock(return_value=values)
    monkeypatch.setattr(gong, 'snapshot_values', snapshot)
    config = SimpleNamespace(observations={'gong': SimpleNamespace(
        backfill=SimpleNamespace(days=2), sources=SimpleNamespace(live=['gong.live'], historical=['gong.archive']))})

    async def check():
        try:
            first = await gong.collect(config, mode='live', now=now)
            assert first['files']['gong']['received'] == 1
            again = await gong.collect(config, mode='backfill', now=now)
            assert again['files']['gong']['retained'] == 1
            assert snapshot.call_count == 1
            async with get_session_factory()() as session:
                assert await gong.load_gong_features(session, observed) is None
                frame = await gong.load_gong_features(session, now)
                assert frame.features == {'field': 1.} and frame.available_at == received
        finally:
            await dispose_engine()
    asyncio.run(check())
    with runtime(dsns, 'clio', urls) as conn:
        row = conn.execute('SELECT fits_gzip,available_at FROM clio.gong_snapshot').fetchone()
        assert gzip.decompress(row[0]) == b'original' and row[1] == received
