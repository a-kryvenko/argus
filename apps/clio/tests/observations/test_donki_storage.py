import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from clio.db.models.file import File
from clio.db.models.file_observation import FileObservation
from clio.ingestion.extras import donki
from clio.ingestion.utils.file_observations import save
from clio.services.solar_wind_events import stored_event_inputs

NOW = datetime(2026, 11, 3, 12, tzinfo=UTC)
METADATA = dict(request_start=NOW-timedelta(days=30), request_end=NOW+timedelta(days=1))


def record(endpoint, rows, *, minute=0):
    return donki.record_poll(endpoint, raw=json.dumps(rows).encode(),
        observed_at=NOW+timedelta(minutes=minute), metadata=METADATA)


def counts(storage):
    with storage.engine.connect() as connection:
        result = tuple(connection.scalar(select(func.count()).select_from(FileObservation).where(
            FileObservation.details['kind'].as_string() == kind)) for kind in ('poll', 'event', 'arrival'))
        result += (connection.scalar(select(func.count()).select_from(FileObservation).where(
            FileObservation.source_id.is_(None))), connection.scalar(select(func.count()).select_from(File)))
        assert connection.scalar(select(func.count()).select_from(FileObservation)) == sum(result[:4])
        return result


def cme(identity='cme1', speed=800):
    return dict(activityID=identity, startTime='2026-11-02T00:00Z', unused='discard-me',
        cmeAnalyses=[dict(speed=speed, isMostAccurate=True)])


def test_repeated_identical_polls_do_not_grow_the_database_or_object_store(storage, metadata_session):
    identities = None
    for minute in range(20):
        record('CME', [cme()], minute=minute)
        assert counts(storage) == (1, 1, 0, 0, 1)
        with storage.engine.connect() as connection:
            current = set(connection.execute(select(FileObservation.id)).scalars())
        identities = current if identities is None else identities
        assert current == identities
    assert len(storage.objects) == 1
    inputs = asyncio.run(stored_event_inputs(metadata_session, NOW+timedelta(minutes=20)))
    assert [e.event_id for e in inputs.events] == ['cme1']
    assert inputs.polls[0].received_at == NOW+timedelta(minutes=19)
    assert inputs.latest_attempts['CME']['last_success_at'] == (NOW+timedelta(minutes=19)).isoformat()


def test_file_catalog_returns_the_current_original_once(storage, metadata_session):
    from clio.routers.observation_files import load_observation_files
    record('CME', [cme(), cme('cme2')])
    originals = asyncio.run(load_observation_files(metadata_session, observation='donki.CME',
        start=NOW-timedelta(days=2), end=NOW+timedelta(days=1)))
    assert len(originals) == 1
    assert originals[0].source_product == 'CME'


@pytest.mark.parametrize('raw', [b'broken json', b'{}', b'[NaN]'])
def test_invalid_download_is_rejected_before_upload(storage, raw):
    with pytest.raises(ValueError):
        donki.record_poll('CME', raw=raw, metadata={})
    assert not storage.objects and counts(storage) == (0, 0, 0, 0, 0)


def test_only_normalized_data_is_stored_and_reads_need_no_originals(storage, metadata_session, monkeypatch):
    record('CME', [cme()])
    with storage.engine.connect() as connection:
        event = connection.execute(select(FileObservation.details['data']).where(
            FileObservation.details['kind'].as_string() == 'event')).scalar_one()
    assert 'discard-me' not in json.dumps(event)
    assert event['initial_speed_km_s'] == 800
    monkeypatch.setattr(donki.datamanager, 'read_file', lambda *a, **k: pytest.fail('Fetched an original'))
    # Other endpoints also update from normalized rows, without re-reading old raw responses.
    record('HSS', [])
    inputs = asyncio.run(stored_event_inputs(metadata_session, NOW))
    assert inputs.events[0].initial_speed_km_s == 800
    assert {p.endpoint: p.record_count for p in inputs.polls} == {'CME': 1, 'HSS': 0}
    assert all('records' not in p for p in inputs.model_dump()['polls'])


def test_simulation_ids_are_unique_and_endpoint_deletions_preserve_other_contributions(storage, metadata_session):
    nested = dict(link='run1', modelCompletionTime='2026-11-03T10:00Z',
        estimatedShockArrivalTime='2026-11-04T12:00Z')
    event = cme(speed=900)
    event['cmeAnalyses'][0]['enlilList'] = [nested]
    cme_poll = record('CME', [event])
    simulation_poll = record('WSAEnlilSimulations', [nested])
    inputs = asyncio.run(stored_event_inputs(metadata_session, NOW))
    assert counts(storage) == (2, 1, 1, 0, 2)
    assert inputs.enlil_arrivals[0].cme_ids == ['cme1']
    assert {r.poll_id for r in inputs.enlil_arrivals[0].receipts} == {cme_poll, simulation_poll}
    record('WSAEnlilSimulations', [{**nested, 'estimatedShockArrivalTime': None}], minute=1)
    inputs = asyncio.run(stored_event_inputs(metadata_session, NOW+timedelta(minutes=1)))
    assert inputs.enlil_arrivals[0].earth_arrival_at is None
    assert inputs.enlil_arrivals[0].cme_ids == ['cme1']
    assert counts(storage) == (2, 1, 1, 0, 2)
    record('WSAEnlilSimulations', [], minute=2)
    inputs = asyncio.run(stored_event_inputs(metadata_session, NOW+timedelta(minutes=2)))
    assert inputs.enlil_arrivals[0].earth_arrival_at is not None  # still present in CME
    assert len(inputs.enlil_arrivals[0].receipts) == 1
    record('WSAEnlilSimulations', [nested], minute=3)
    record('CME', [], minute=4)
    inputs = asyncio.run(stored_event_inputs(metadata_session, NOW+timedelta(minutes=4)))
    assert inputs.events == [] and inputs.enlil_arrivals[0].cme_ids == []
    assert counts(storage) == (2, 0, 1, 0, 2)
    record('WSAEnlilSimulations', [], minute=5)
    assert counts(storage) == (2, 0, 0, 0, 2)


def test_corrections_update_ids_and_full_responses_remove_missing_events(storage, metadata_session):
    record('CME', [cme(), cme('cme2')])
    record('HSS', [dict(hssID='hss1', eventTime='2026-11-02T00:00Z', location='L1')])
    record('CME', [cme(speed=1000)], minute=1)
    inputs = asyncio.run(stored_event_inputs(metadata_session, NOW+timedelta(minutes=1)))
    assert {e.event_id: e.initial_speed_km_s for e in inputs.events} == {'cme1': 1000, 'hss1': None}
    assert counts(storage) == (2, 2, 0, 0, 2)
    record('CME', [], minute=2)
    inputs = asyncio.run(stored_event_inputs(metadata_session, NOW+timedelta(minutes=2)))
    assert [e.event_id for e in inputs.events] == ['hss1']
    assert {p.endpoint: p.record_count for p in inputs.polls} == {'CME': 0, 'HSS': 1}
    assert inputs.polls[0].received_at == NOW+timedelta(minutes=2)
    assert counts(storage) == (2, 1, 0, 0, 2)


def test_upload_failure_rolls_back_event_changes_deletions_and_success_time(storage, metadata_session, monkeypatch):
    record('CME', [cme(), cme('cme2')])
    def fail(*args, **kwargs):
        raise OSError('S3 unavailable')
    monkeypatch.setattr(donki.datamanager, 'put_immutable_file', fail)
    with pytest.raises(OSError):
        record('CME', [cme(speed=1000)], minute=1)
    inputs = asyncio.run(stored_event_inputs(metadata_session, NOW+timedelta(minutes=1)))
    assert {e.event_id: e.initial_speed_km_s for e in inputs.events} == {'cme1': 800, 'cme2': 800}
    assert inputs.polls[0].received_at == NOW
    assert counts(storage) == (1, 2, 0, 0, 1)


def test_out_of_order_poll_cannot_replace_newer_data(storage, metadata_session):
    latest = record('CME', [cme(speed=1000)], minute=2)
    assert record('CME', [], minute=1) == latest
    inputs = asyncio.run(stored_event_inputs(metadata_session, NOW+timedelta(minutes=3)))
    assert inputs.events[0].initial_speed_km_s == 1000
    assert inputs.polls[0].received_at == NOW+timedelta(minutes=2)
    assert counts(storage) == (1, 1, 0, 0, 1) and len(storage.objects) == 1


def test_legacy_snapshots_are_imported_once_and_removed_without_touching_other_files(storage, metadata_session, tmp_path):
    original = tmp_path / 'original.json'
    for minute, speed in ((0, 700), (1, 900)):
        original.write_text(json.dumps([cme(speed=speed)]))
        save(original, observation='donki.CME', observed_at=NOW+timedelta(minutes=minute),
            source_product='CME', details=dict(METADATA, records=[cme(speed=speed)]))
    profile = tmp_path / 'original.nc'
    profile.write_bytes(b'legacy profile')
    save(profile, observation='enlil', observed_at=NOW, source_product='noaa.enlil',
        details=dict(payload=dict(points=[dict(speed_km_s=900)])))
    unrelated = save(profile, observation='gong', observed_at=NOW, source_product='gong.live')
    record('HSS', [], minute=2)
    inputs = asyncio.run(stored_event_inputs(metadata_session, NOW+timedelta(minutes=2)))
    assert inputs.events[0].initial_speed_km_s == 900
    assert counts(storage) == (2, 1, 0, 1, 3)
    with storage.engine.connect() as connection:
        assert connection.execute(select(FileObservation.id).where(FileObservation.source_id.is_(None))).scalar_one() == unrelated['id']
    assert len(storage.objects) == 5  # originals are preserved; no DB snapshot history remains


def test_corrupt_legacy_original_rolls_back_import_and_keeps_existing_receipts(storage, metadata_session, tmp_path):
    original = tmp_path / 'original.json'
    original.write_text(json.dumps([cme()]))
    receipt = save(original, observation='donki.CME', observed_at=NOW, source_product='CME',
        details=dict(METADATA, records=[cme()]))
    location = 'argus-private', receipt['file']['key']
    storage.objects[location] = (b'[]', *storage.objects[location][1:])
    with pytest.raises(ValueError, match='checksum'):
        record('HSS', [])
    assert counts(storage) == (0, 0, 0, 1, 1) and len(storage.objects) == 1
    # A fresh successful response for that endpoint can recover without its old original.
    record('CME', [cme(speed=1000)], minute=1)
    assert counts(storage) == (1, 1, 0, 0, 1)
