"""PostgreSQL catalogs read saved references without downloading objects."""
import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest
from common.data import datamanager
from tests.support.database import database


@pytest.mark.parametrize('kind', ['gong', 'goes'])
def test_snapshot_catalog_reads_metadata_without_object_downloads(file_database, monkeypatch, tmp_path, kind):
    from clio.ingestion.utils.file_observations import save
    from clio.routers.observation_files import load_observation_files
    from clio.db.session import get_db_session
    from common.db.session import dispose_engine
    observed_at = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    path = tmp_path / 'original.fits'
    path.write_bytes(b'original')
    save(path, observation=kind, observed_at=observed_at, source_product=kind+'.live')

    async def check():
        try:
            monkeypatch.setattr(datamanager, 'read_reference', Mock(side_effect=AssertionError('No storage reads in a catalog')))
            async with get_db_session() as session:
                refs = await load_observation_files(session, start=observed_at,
                    end=observed_at+timedelta(hours=1), observation=kind)
                assert len(refs) == 1 and refs[0].file.bucket == 'argus-private'
                assert not await load_observation_files(session, start=observed_at+timedelta(hours=1),
                    end=observed_at+timedelta(hours=2), observation=kind)
        finally:
            await dispose_engine()
    asyncio.run(check())
