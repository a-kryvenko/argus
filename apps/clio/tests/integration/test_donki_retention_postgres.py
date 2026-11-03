"""Exercise DONKI poll transitions, compaction and object removal on PostgreSQL."""
import asyncio
from datetime import UTC, datetime, timedelta
import hashlib
import json
from uuid import uuid4

from sqlalchemy import insert

from tests.support.database import database, migrate


def test_donki_deduplication_preserves_postgres_as_of_reads_and_compacts_legacy_files(database, monkeypatch, object_store):
    _, _, environment = database
    migrate(environment)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    from common.db.session import dispose_engine
    from clio.db.session import get_db_session
    from clio.db.models.file_observation import FileObservation as Receipt
    from clio.observations import files, donki, file_storage, archive
    from clio.observations.solar_wind_events import donki_inputs
    now = datetime.now(UTC).replace(microsecond=0)
    metadata = dict(request_start=now-timedelta(days=30), request_end=now+timedelta(days=1),
                    source_url='https://example.test/CME')
    a = json.dumps([dict(activityID='a', startTime=(now-timedelta(days=1)).isoformat())]).encode()
    b = json.dumps([dict(activityID='b', startTime=(now-timedelta(days=1)).isoformat())]).encode()
    for minute, raw, error in ((-45, a, None), (-30, a, None), (-15, b, None),
                               (0, a, None), (1, None, 'Timeout'), (2, a, None)):
        donki.record_poll('CME', raw=raw, error=error, metadata=metadata, received_at=now+timedelta(minutes=minute))
    assert len(object_store.objects) == 2
    # Simulate already-migrated duplicate objects with the former UUID-based naming.
    for minute in (3, 4):
        identity = uuid4()
        at = now+timedelta(minutes=minute)
        reference = file_storage.publish(a, 'argus-private', archive.remote_key('donki', at, 'CME', identity),
                                         content_type='application/json')
        details = dict(metadata, file_id=str(reference['id']), sha256=hashlib.sha256(a).hexdigest(), records=json.loads(a))
        with files.engine().begin() as connection:
            connection.execute(insert(Receipt).values(id=identity, kind='donki', source_key=str(identity),
                source_product='CME', received_at=at, **archive.file_fields('donki', details)))
    report = donki.cleanup(now+timedelta(minutes=5), 30)
    assert report['deduplicated'] == report['files_deleted'] == 2
    assert len(object_store.objects) == 2

    async def check():
        try:
            for minute, expected, failure in ((-45, 'a', None), (-30, 'a', None), (-15, 'b', None),
                                               (0, 'a', None), (1, 'a', 'Timeout'), (2, 'a', None), (4, 'a', None)):
                async with get_db_session() as session:
                    polls, attempts = await donki_inputs(session, now+timedelta(minutes=minute))
                    assert polls[0].records[0]['activityID'] == expected
                    assert attempts['CME']['error'] == failure
        finally:
            await dispose_engine()
    asyncio.run(check())
