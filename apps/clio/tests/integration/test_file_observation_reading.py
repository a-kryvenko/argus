"""PostgreSQL catalogs use database metadata and S3 references without local files."""
from common.data import datamanager
import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest
from tests.support.database import database, migrate


@pytest.mark.parametrize('kind',['gong','goes'])
def test_snapshot_catalog_reads_metadata_without_object_downloads(database,monkeypatch,object_store,kind):
    _,_,environment = database
    migrate(environment)
    for key,value in environment.items():
        monkeypatch.setenv(key,value)
    from clio.ingestion.utils import archive
    from clio.routers.observation_files import load_observation_files
    from clio.db.models.file_observation import FileObservation
    from clio.db.session import get_db_session
    from common.db.session import dispose_engine
    slot = datetime.now(UTC).replace(minute=0,second=0,microsecond=0)
    record = archive.store_original(kind,slot,slot,kind+'.live',lambda:b'original')
    async def check():
        try:
            async with get_db_session() as session:
                session.add(FileObservation(**archive.observation_values(kind,record)))
                await session.commit()
            monkeypatch.setattr(datamanager,'read_reference',Mock(side_effect=AssertionError('No storage reads in a catalog')))
            async with get_db_session() as session:
                refs = await load_observation_files(session,start=slot,end=slot+timedelta(hours=1),observation=kind)
                assert len(refs) == 1 and refs[0].file.bucket == 'argus-private'
                assert not await load_observation_files(session,start=slot+timedelta(hours=1),end=slot+timedelta(hours=2),observation=kind)
        finally:
            await dispose_engine()
    asyncio.run(check())
