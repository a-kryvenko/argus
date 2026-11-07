"""Normalized DONKI observations and numeric histories in PostgreSQL."""
import asyncio
import json
from datetime import UTC, datetime, timedelta

from tests.support.database import database, migrate


def test_latest_donki_response_replaces_old_events_even_when_empty(file_database):
    from clio.ingestion.extras.donki import record_poll
    from common.db.session import dispose_engine
    from clio.db.session import get_db_session
    from clio.services.solar_wind_events import load_event_inputs
    now = datetime.now(UTC)
    for minute, content in ((0, b'[{"activityID":"cme1","startTime":"2026-01-01T00:00Z"}]'), (1, b'[]')):
        record_poll('CME', raw=content, metadata=dict(request_start=(now-timedelta(days=30)).isoformat(),
            request_end=(now+timedelta(days=1)).isoformat()), observed_at=now+timedelta(minutes=minute))

    async def check():
        try:
            async with get_db_session() as session:
                result = await load_event_inputs(session, now+timedelta(minutes=2))
                assert result.polls[0].records == [] and result.events == []
                assert result.latest_attempts['CME']['received_at'] == (now+timedelta(minutes=1)).isoformat()
        finally:
            await dispose_engine()
    asyncio.run(check())


def test_legacy_metadata_is_pruned_after_normalized_predictions_are_published(file_database, tmp_path, monkeypatch):
    from sqlalchemy import func, select
    from clio.db.models.file_observation import FileObservation
    from clio.db.models.file import File
    from clio.ingestion.extras.donki import record_poll
    from clio.ingestion.utils.file_observations import save
    from clio.services.solar_wind_events import read_event_inputs
    from clio.db.session import get_db_session
    from common.data import datamanager
    from common.db.session import dispose_engine

    now = datetime.now(UTC)
    metadata = dict(request_start=now-timedelta(days=30), request_end=now+timedelta(days=1))
    cme = dict(activityID='cme1', startTime=(now-timedelta(days=1)).isoformat(),
        cmeAnalyses=[dict(speed=900)])
    path = tmp_path / 'original.json'
    path.write_text(json.dumps([cme]))
    save(path, observation='donki.CME', observed_at=now, source_product='CME',
        details=dict(metadata, records=[cme]))
    record_poll('WSAEnlilSimulations', metadata=metadata, observed_at=now,
        raw=json.dumps([dict(link='run1', cmeInputs=[dict(CMEID='cme1')],
            modelCompletionTime=(now-timedelta(hours=1)).isoformat(),
            estimatedShockArrivalTime=(now+timedelta(days=1)).isoformat())]).encode())
    with file_database.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(FileObservation)) == 4
        assert connection.scalar(select(func.count()).select_from(File)) == 2
        assert connection.scalar(select(func.count()).select_from(FileObservation).where(
            FileObservation.source_id.is_(None))) == 0
    def unexpected_read(*args, **kwargs):
        raise AssertionError('API read must not fetch originals')
    monkeypatch.setattr(datamanager, 'read_file', unexpected_read)

    async def check():
        try:
            async with get_db_session() as session:
                result = await read_event_inputs(session, now+timedelta(seconds=1))
                assert result.events[0].initial_speed_km_s == 900
                assert result.enlil_arrivals[0].cme_ids == ['cme1']
                assert len(result.polls) == 2
                assert all(p.record_count == 1 for p in result.polls)
                assert all('records' not in p for p in result.model_dump()['polls'])
        finally:
            await dispose_engine()
    asyncio.run(check())


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


def test_closed_hourly_speed_history_includes_late_receipts_and_prefers_native_data(database, monkeypatch, object_store):
    _, _, environment = database
    migrate(environment)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    from clio.db.models.measurement import Measurement
    from common.db.session import dispose_engine
    from clio.db.session import get_db_session
    from clio.services.solar_wind_events import hourly_speeds
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    first = now-timedelta(days=27)
    second = first+timedelta(hours=1)

    async def check():
        try:
            async with get_db_session() as session:
                for hour in (first, second):
                    session.add(Measurement(metric='v', observed_at=hour, received_at=hour+timedelta(hours=2),
                        value=400, source_product='omni.hourly'))
                    minutes = 30 if hour == first else 10
                    for minute in range(1, minutes+1):
                        session.add(Measurement(metric='v', observed_at=hour+timedelta(minutes=minute),
                            received_at=hour+timedelta(hours=1), value=600, source_product='swpc.rtsw_plasma'))
                session.add(Measurement(metric='v', observed_at=now-timedelta(hours=2),
                    received_at=now+timedelta(minutes=1), value=900, source_product='omni.hourly'))
                session.add(Measurement(metric='v', observed_at=now,
                    received_at=now, value=900, source_product='swpc.rtsw_plasma'))
                await session.commit()
            async with get_db_session() as session:
                result = await hourly_speeds(session, now)
                assert len(result) == 3
                assert result[0].value == 600 and result[0].sample_count == 30
                assert result[1].value == 400 and result[1].source_product == 'omni.hourly'
                assert result[2].value == 900 and result[2].available_at > now
                assert all(r.interval_end <= now for r in result)
        finally:
            await dispose_engine()
    asyncio.run(check())
