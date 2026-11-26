"""The GONG collector stores original objects and metadata without local archives."""
import asyncio
from datetime import UTC, datetime
from io import BytesIO
from unittest.mock import Mock
from astropy.io import fits
import numpy as np
import pandas as pd
from sqlalchemy import select
from clio.db.models.file_observation import FileObservation


def test_gong_collection_writes_s3_reference_and_retains_existing_snapshot(file_database,monkeypatch):
    from clio.config import ClioObservations, load_observation_config
    from clio.ingestion.extras import gong
    from clio.providers import gong as provider
    from common.db.session import dispose_engine
    now = datetime.now(UTC).replace(minute=30,second=0,microsecond=0)
    observed = now.replace(minute=0)
    monkeypatch.setattr(provider,'candidates',Mock(return_value=pd.DataFrame([dict(timestamp=observed,gong_url='https://example.test/gong.fits')])) )
    picture = fits.PrimaryHDU(np.ones((2, 2), dtype='float32'))
    picture.header['TELESCOP'] = 'GONG'
    picture.header['IMTYPE'] = 'MAGNETIC'
    picture.header['MAPDATE'] = observed.date().isoformat()
    picture.header['MAPTIME'] = observed.strftime('%H:%M:%S')
    contents = BytesIO()
    picture.writeto(contents)
    download = Mock(return_value=contents.getvalue())
    monkeypatch.setattr(provider,'download',download)
    async def check():
        try:
            config = ClioObservations.model_validate({'observations': {'gong': {**load_observation_config().observations['gong'].model_dump(), 'backfill_days': 60, 'retention_days': 60, 'freshness_minutes': 120}}})
            first = await gong.collect(config.observations['gong'], source='gong.live', mode='live', now=now)
            assert first['received'] == 1 and first['failed'] == 0
            assert first['latest_observed_at'] == observed.isoformat()
            await gong.collect(config.observations['gong'], source='gong.live', mode='live', now=now)
            assert download.call_count == 1
            with file_database.connect() as connection:
                row = connection.execute(select(FileObservation.file_id, FileObservation.details)).one()
                assert row[0] is not None
                assert row[1]['source_url'] == 'https://example.test/gong.fits'
        finally:
            await dispose_engine()
    asyncio.run(check())
