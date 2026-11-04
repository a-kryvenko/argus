import asyncio
from datetime import UTC, datetime, timedelta
import hashlib
import json
from types import SimpleNamespace
from uuid import uuid4

from botocore.exceptions import ClientError
import pytest
from sqlalchemy import select, insert
from sqlalchemy.orm import Session

from clio.db.models.file import File
from clio.db.models.file_observation import FileObservation as Receipt
from clio.ingestion.extras import donki
from clio.ingestion.utils import archive
from clio.routers.solar_wind_events import donki_inputs

NOW = datetime(2026, 11, 3, 12, tzinfo=UTC)
A, B = b'[{"activityID":"a"}]', b'[{"activityID":"b"}]'


def metadata(at=NOW):
    return dict(request_start=at.replace(hour=0, minute=0, second=0, microsecond=0)-timedelta(days=30),
                request_end=at.replace(hour=0, minute=0, second=0, microsecond=0)+timedelta(days=1), source_url='https://example.test/CME')


def poll(raw=A, at=NOW):
    return donki.record_poll('CME', raw=raw, received_at=at, metadata=metadata(at))


def receipts(storage):
    with storage.engine.connect() as connection:
        return [dict(row) for row in connection.execute(select(Receipt.__table__).order_by(Receipt.received_at)).mappings()]


def inputs(storage):
    with Session(storage.engine) as session:
        class AsyncReader:
            async def scalars(self, statement):
                rows = [SimpleNamespace(**{column.name: getattr(row, column.name) for column in Receipt.__table__.columns})
                        for row in session.scalars(statement).all()]
                return SimpleNamespace(all=lambda: rows)
            async def execute(self, statement):
                rows = [SimpleNamespace(**row) for row in session.execute(statement).mappings()]
                return SimpleNamespace(all=lambda: rows)
        return asyncio.run(donki_inputs(AsyncReader()))


def seed_legacy(storage, raw, at, *, endpoint='CME'):
    identity = uuid4()
    details = dict(metadata(at), sha256=hashlib.sha256(raw).hexdigest(), records=json.loads(raw))
    reference = storage.publish(raw, 'argus-private', archive.remote_key('donki', at, endpoint, identity),
                                     content_type='application/json')
    details['file_id'] = str(reference['id'])
    with storage.engine.begin() as connection:
        connection.execute(insert(Receipt).values(id=identity, kind='donki', source_product=endpoint,
            source_key=str(identity), received_at=at, **archive.file_fields('donki', details)))
    return identity, reference


def test_unchanged_poll_records_successful_checks_without_duplicate_files(storage):
    first = poll()
    writes = storage.client.put_object.call_count
    poll(at=NOW+timedelta(minutes=15))
    poll(at=NOW+timedelta(minutes=45))
    rows = receipts(storage)
    assert [row['kind'] for row in rows] == ['donki', 'donki_poll', 'donki_poll']
    assert rows[1]['details'] is None and rows[2]['details'] is None
    assert rows[0]['file_id'] == rows[1]['file_id'] == rows[2]['file_id']
    assert storage.client.put_object.call_count == writes == 1
    versions, attempts = inputs(storage)
    assert versions[0].poll_id == first
    assert versions[0].received_at == NOW
    assert attempts['CME']['error'] is None
    assert datetime.fromisoformat(attempts['CME']['last_success_at']) == NOW+timedelta(minutes=45)
    assert datetime.fromisoformat(attempts['CME']['received_at']) == NOW+timedelta(minutes=45)


def test_changes_reversions_and_empty_responses_return_latest_stored_version(storage):
    for minute, raw in [(0, A), (15, B), (30, A), (45, b'[]')]:
        poll(raw, NOW+timedelta(minutes=minute))
        versions, _ = inputs(storage)
        assert versions[0].records == json.loads(raw)
        assert versions[0].received_at == NOW+timedelta(minutes=minute)
    rows = receipts(storage)
    assert len(rows) == 4 and all(row['kind'] == 'donki' for row in rows)
    assert rows[0]['file_id'] == rows[2]['file_id']
    assert len(storage.objects) == 3


def test_new_request_window_reuses_object_but_keeps_correct_coverage(storage):
    poll(b'[]')
    poll(b'[]', NOW+timedelta(days=1))
    assert len(storage.objects) == 1
    assert [row['kind'] for row in receipts(storage)] == ['donki', 'donki']
    versions, _ = inputs(storage)
    assert versions[0].request_start == metadata(NOW+timedelta(days=1))['request_start']


def test_cleanup_merges_existing_duplicates_and_preserves_state_transitions(storage):
    first, reference = seed_legacy(storage, A, NOW-timedelta(minutes=45))
    duplicate, _ = seed_legacy(storage, A, NOW-timedelta(minutes=30))
    changed, _ = seed_legacy(storage, B, NOW-timedelta(minutes=15))
    reverted, _ = seed_legacy(storage, A, NOW)
    report = donki.cleanup(NOW, 30)
    assert report == dict(expired=0, deduplicated=1, files_deleted=2, failed=0)
    rows = receipts(storage)
    assert rows[1]['kind'] == 'donki_poll' and rows[1]['details'] is None
    assert rows[0]['file_id'] == rows[1]['file_id'] == rows[3]['file_id'] == reference['id']
    assert len(storage.objects) == 2
    assert inputs(storage)[0][0].poll_id == reverted
    assert donki.cleanup(NOW, 30) == dict(expired=0, deduplicated=0, files_deleted=0, failed=0)


def test_retention_preserves_boundary_snapshot_and_latest_attempt(storage):
    seed_legacy(storage, A, NOW-timedelta(days=45))
    boundary, ref = seed_legacy(storage, B, NOW-timedelta(days=40))
    recent = poll(b'[]', NOW)
    report = donki.cleanup(NOW, 30)
    assert report['expired'] == report['files_deleted'] == 1
    assert len(receipts(storage)) == 2
    assert boundary in {row['id'] for row in receipts(storage)}
    versions, attempts = inputs(storage)
    assert versions[0].poll_id == recent and attempts['CME']['error'] is None
    assert storage.objects['argus-private', ref['key']][0] == B


def test_delete_failure_keeps_orphan_registered_for_retry(storage):
    reference = storage.publish(A, 'argus-private', 'observations/DONKI/orphan.json', content_type='application/json')
    delete = storage.client.delete_object.side_effect
    storage.client.delete_object.side_effect = ClientError({'Error': {'Code': 'AccessDenied'}}, 'DeleteObject')
    assert donki.cleanup(NOW, 30)['failed'] == 1
    assert storage.files(id=reference['id'])
    storage.client.delete_object.side_effect = delete
    assert donki.cleanup(NOW, 30)['files_deleted'] == 1
    assert not storage.files(id=reference['id'])


def test_cleanup_only_deletes_unreferenced_donki_objects_and_obeys_batch_limit(storage):
    poll()
    storage.publish(A, 'argus-private', 'images/SDO/keep.npz', content_type='application/octet-stream')
    storage.publish(A, 'argus-private', 'observations/ENLIL/keep.nc', content_type='application/octet-stream')
    for n in range(3):
        storage.publish(A, 'argus-private', f'observations/DONKI/unused-{n}.json', content_type='application/json')
    assert donki.cleanup(NOW, 30, max_files=2)['files_deleted'] == 2
    assert len(storage.objects) == 4
    assert donki.cleanup(NOW, 30)['files_deleted'] == 1
    assert len(storage.objects) == 3


def test_cleanup_and_collection_share_a_lock(storage):
    from clio.ingestion.extras import donki as collector
    with donki.lock():
        assert donki.cleanup(NOW, 30)['busy']
        assert asyncio.run(collector.collect(SimpleNamespace(lookback_days=30), now=NOW)) is False


def test_retention_preserves_successful_check_at_boundary(storage):
    seed_legacy(storage, A, NOW-timedelta(days=45))
    checked = NOW-timedelta(days=32)
    donki.record_poll('CME', raw=A, received_at=checked, metadata=metadata(NOW-timedelta(days=45)))
    donki.cleanup(NOW, 30)
    versions, attempts = inputs(storage)
    assert versions[0].received_at == NOW-timedelta(days=45)
    assert attempts['CME']['error'] is None
    assert datetime.fromisoformat(attempts['CME']['last_success_at']) == checked


@pytest.mark.parametrize('raw', [b'broken json', b'{}', b'[NaN]'])
def test_invalid_response_is_rejected_before_observation_or_upload(storage, raw):
    first = poll()
    with pytest.raises(ValueError):
        poll(raw, NOW+timedelta(minutes=15))
    assert len(receipts(storage)) == len(storage.objects) == 1
    assert inputs(storage)[0][0].poll_id == first


def test_failed_fetch_is_not_stored_and_other_endpoints_continue(storage, monkeypatch):
    from unittest.mock import Mock
    from clio.ingestion.extras import donki as collector
    monkeypatch.setattr(collector.donki, 'ENDPOINTS', ('CME', 'HSS'))
    fetch = Mock(side_effect=[OSError('timeout'), (b'[]', [])])
    monkeypatch.setattr(collector.donki, 'fetch', fetch)
    with pytest.raises(RuntimeError, match='CME'):
        asyncio.run(collector.collect(SimpleNamespace(lookback_days=30), now=NOW))
    assert fetch.call_count == 2
    rows = receipts(storage)
    assert len(rows) == 1 and rows[0]['source_product'] == 'HSS'
