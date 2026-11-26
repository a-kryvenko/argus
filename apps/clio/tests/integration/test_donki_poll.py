"""Normalized DONKI observations in PostgreSQL."""
import json
from datetime import UTC, datetime, timedelta


def read_event_inputs(client):
    response = client.get('/internal/v1/observations/solar-wind-events/inputs')
    assert response.status_code == 200, response.text
    return response.json()


def test_latest_donki_response_replaces_old_events_even_when_empty(file_api):
    from clio.ingestion.extras.donki import record_poll
    now = datetime.now(UTC)
    for minute, content in ((0, b'[{"activityID":"cme1","startTime":"2026-01-01T00:00Z"}]'), (1, b'[]')):
        record_poll('CME', raw=content, metadata=dict(request_start=(now-timedelta(days=30)).isoformat(),
            request_end=(now+timedelta(days=1)).isoformat()), observed_at=now+timedelta(minutes=minute))

    result = read_event_inputs(file_api)
    assert result['polls'][0]['record_count'] == 0 and result['events'] == []
    assert result['latest_attempts']['CME']['received_at'] == (now+timedelta(minutes=1)).isoformat()


def test_legacy_metadata_is_pruned_after_normalized_predictions_are_published(file_database, file_api, tmp_path):
    from sqlalchemy import func, select
    from clio.db.models.file_observation import FileObservation
    from clio.db.models.file import File
    from clio.ingestion.extras.donki import record_poll, restore_legacy_polls
    from clio.ingestion.utils.file_observations import save

    now = datetime.now(UTC)
    metadata = dict(request_start=now-timedelta(days=30), request_end=now+timedelta(days=1))
    cme = dict(activityID='cme1', startTime=(now-timedelta(days=1)).isoformat(),
        cmeAnalyses=[dict(speed=900)])
    path = tmp_path / 'original.json'
    path.write_text(json.dumps([cme]))
    save(path, observation='donki.CME', observed_at=now, source_product='CME',
        details=dict(metadata, records=[cme]))
    assert restore_legacy_polls()['status'] == 'complete'
    record_poll('WSAEnlilSimulations', metadata=metadata, observed_at=now,
        raw=json.dumps([dict(link='run1', cmeInputs=[dict(CMEID='cme1')],
            modelCompletionTime=(now-timedelta(hours=1)).isoformat(),
            estimatedShockArrivalTime=(now+timedelta(days=1)).isoformat())]).encode())
    with file_database.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(FileObservation)) == 4
        assert connection.scalar(select(func.count()).select_from(File)) == 2
        assert connection.scalar(select(func.count()).select_from(FileObservation).where(
            FileObservation.source_id.is_(None))) == 0
    result = read_event_inputs(file_api)
    assert result['events'][0]['initial_speed_km_s'] == 900
    assert result['enlil_arrivals'][0]['cme_ids'] == ['cme1']
    assert len(result['polls']) == 2
    assert all(poll['record_count'] == 1 and 'records' not in poll for poll in result['polls'])


def test_failed_poll_is_distinct_from_last_success(file_database, file_api):
    from clio.ingestion.extras.donki import record_attempt, record_poll

    now = datetime.now(UTC)
    later = now + timedelta(minutes=1)
    record_poll('CME', raw=b'[]', observed_at=now,
                metadata=dict(request_start=now-timedelta(days=30), request_end=now+timedelta(days=1)))
    with file_database.begin() as connection:
        record_attempt(connection, 'CME', later, 'Timeout')
        record_attempt(connection, 'HSS', later, 'ConnectionError')
        record_attempt(connection, 'CME', now, 'OutdatedFailure')
    attempts = read_event_inputs(file_api)['latest_attempts']
    assert attempts['CME'] == {'received_at': later.isoformat(), 'last_success_at': now.isoformat(), 'error': 'Timeout'}
    assert attempts['HSS'] == {'received_at': later.isoformat(), 'last_success_at': None, 'error': 'ConnectionError'}


def test_repeated_polls_upsert_unique_ids_and_failed_writes_roll_back(file_database, monkeypatch):
    import pytest
    from sqlalchemy import func, select
    from clio.db.models.file import File
    from clio.db.models.file_observation import FileObservation
    from clio.ingestion.extras import donki as collector

    now = datetime.now(UTC)
    metadata = dict(request_start=now-timedelta(days=30), request_end=now+timedelta(days=1))
    def record(endpoint, rows, minute):
        return collector.record_poll(endpoint, raw=json.dumps(rows).encode(), metadata=metadata,
            observed_at=now+timedelta(minutes=minute))
    def successful_poll():
        return select(FileObservation.observed_at).where(
            FileObservation.observation == 'donki.CME', FileObservation.source_id == '')
    cme = dict(activityID='cme1', startTime=(now-timedelta(days=1)).isoformat(), cmeAnalyses=[dict(speed=900)])
    for minute in range(20):
        record('CME', [cme, {**cme, 'activityID': 'cme2'}], minute)
    record('HSS', [dict(hssID='hss1', eventTime=(now-timedelta(days=1)).isoformat(), location='L1')], 20)
    with file_database.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(FileObservation)) == 5
        assert connection.scalar(select(func.count()).select_from(File)) == 2
        assert connection.scalar(successful_poll()) == now+timedelta(minutes=19)
    original_upsert = collector.upsert
    def fail_after_deletion(connection, values):
        if values['source_id'] == 'cme1':
            raise RuntimeError('injected write failure')
        return original_upsert(connection, values)
    monkeypatch.setattr(collector, 'upsert', fail_after_deletion)
    with pytest.raises(RuntimeError):
        record('CME', [cme], 21)
    with file_database.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(FileObservation)) == 5
        assert connection.scalar(successful_poll()) == now+timedelta(minutes=19)
    monkeypatch.setattr(collector, 'upsert', original_upsert)
    record('CME', [], 22)
    with file_database.connect() as connection:
        assert connection.execute(select(FileObservation.source_id).where(
            FileObservation.details['kind'].as_string() == 'event')).scalars().all() == ['hss1']
        assert connection.scalar(select(func.count()).select_from(FileObservation)) == 3
        assert connection.scalar(select(func.count()).select_from(File)) == 2
        assert connection.scalar(successful_poll()) == now+timedelta(minutes=22)
