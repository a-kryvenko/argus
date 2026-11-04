"""The GONG collector stores original objects and metadata without local archives."""
import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock
import pandas as pd
from tests.support.database import database,migrate,runtime


def test_gong_collection_writes_s3_reference_and_retains_existing_snapshot(database,monkeypatch,object_store):
    dsns,urls,environment = database
    migrate(environment)
    for key,value in environment.items():
        monkeypatch.setenv(key,value)
    from clio.config import ClioObservations, load_observation_config
    from clio.ingestion.extras import gong
    from clio.providers import gong as provider
    from common.db.session import dispose_engine
    now = datetime.now(UTC).replace(minute=30,second=0,microsecond=0)
    observed = now.replace(minute=0)
    monkeypatch.setattr(provider,'candidates',Mock(return_value=pd.DataFrame([dict(timestamp=observed,gong_url='https://example.test/gong.fits')])) )
    download = Mock(return_value=b'original FITS')
    monkeypatch.setattr(provider,'download',download)
    async def check():
        try:
            config = ClioObservations.model_validate({'observations': {'gong': {**load_observation_config().observations['gong'].model_dump(), 'backfill_days': 60, 'retention_days': 60, 'freshness_minutes': 120}}})
            first = await gong.collect(config,mode='live',now=now)
            assert first['files']['gong']['received'] == 1
            await gong.collect(config,mode='live',now=now)
            assert download.call_count == 1
            with runtime(dsns,'clio',urls) as connection:
                row = connection.execute('SELECT file_id,details FROM clio.file_observation').fetchone()
                assert row[0] is not None
                assert row[1]['source_url'] == 'https://example.test/gong.fits'
        finally:
            await dispose_engine()
    asyncio.run(check())
