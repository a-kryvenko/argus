"""Exercise DONKI poll transitions, compaction and object removal on PostgreSQL."""
import asyncio
from datetime import UTC, datetime, timedelta
import hashlib
import json
from uuid import uuid4

from sqlalchemy import insert

from tests.support.database import database, migrate


def test_donki_deduplication_preserves_latest_poll_and_compacts_legacy_files(database, monkeypatch, object_store):
    _, _, environment = database
    migrate(environment)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    from common.db.session import dispose_engine
    from clio.db import session as db
    from clio.db.session import get_db_session
    from clio.db.models.file_observation import FileObservation as Receipt
    from clio.ingestion.extras import donki
    from clio.ingestion.utils import archive
    from clio.routers.solar_wind_events import donki_inputs
    now = datetime.now(UTC).replace(microsecond=0)
    metadata = dict(request_start=now-timedelta(days=30), request_end=now+timedelta(days=1),
                    source_url='https://example.test/CME')
    a = json.dumps([dict(activityID='a', startTime=(now-timedelta(days=1)).isoformat())]).encode()
    b = json.dumps([dict(activityID='b', startTime=(now-timedelta(days=1)).isoformat())]).encode()
    for minute, raw in ((-45, a), (-30, a), (-15, b), (0, a), (2, a)):
        donki.record_poll('CME', raw=raw, metadata=metadata, received_at=now+timedelta(minutes=minute))
    assert len(object_store.objects) == 2
    # Simulate already-migrated duplicate objects with the former UUID-based naming.
    for minute in (3, 4):
        identity = uuid4()
        at = now+timedelta(minutes=minute)
        reference = object_store.publish(a, 'argus-private', archive.remote_key('donki', at, 'CME', identity),
                                         content_type='application/json')
        details = dict(metadata, file_id=str(reference['id']), sha256=hashlib.sha256(a).hexdigest(), records=json.loads(a))
        with db.get_sync_engine().begin() as connection:
            connection.execute(insert(Receipt).values(id=identity, kind='donki', source_key=str(identity),
                source_product='CME', received_at=at, **archive.file_fields('donki', details)))
    report = donki.cleanup(now+timedelta(minutes=5), 30)
    assert report['deduplicated'] == report['files_deleted'] == 2
    assert len(object_store.objects) == 2

    async def check():
        try:
            async with get_db_session() as session:
                polls, attempts = await donki_inputs(session)
                assert polls[0].records[0]['activityID'] == 'a'
                assert attempts['CME']['error'] is None
                assert attempts['CME']['received_at'] == (now+timedelta(minutes=4)).isoformat()
        finally:
            await dispose_engine()
    asyncio.run(check())
