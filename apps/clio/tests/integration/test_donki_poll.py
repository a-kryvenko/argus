"""Latest receipts, empty replacement and successful-poll history in PostgreSQL."""
import asyncio
import gzip
import hashlib
from datetime import UTC, datetime, timedelta

from tests.support.database import database, migrate


def test_latest_donki_response_replaces_old_events_even_when_empty(database, monkeypatch, object_store):
    _, _, environment = database
    migrate(environment)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    from clio.db.models.file_observation import FileObservation
    from clio.ingestion.utils.archive import store_receipt
    from common.db.session import dispose_engine
    from clio.db.session import get_db_session
    from clio.routers.solar_wind_events import load_event_inputs
    now = datetime.now(UTC)
    raw = b'[{"activityID":"cme1","startTime":"2026-01-01T00:00Z"}]'

    async def check():
        try:
            async with get_db_session() as session:
                for minute, payload in ((0, [{'activityID': 'cme1', 'startTime': '2026-01-01T00:00Z'}]), (1, [])):
                    content = raw if payload else b'[]'
                    values = store_receipt('donki', 'CME', str(minute),
                        received_at=now+timedelta(minutes=minute), raw=content,
                        metadata=dict(request_start=now-timedelta(days=30), request_end=now+timedelta(days=1),
                            source_url='https://ccmc.gsfc.nasa.gov/DONKI-API/get/CME'))
                    session.add(FileObservation(**values))
                await session.commit()
            async with get_db_session() as session:
                second = await load_event_inputs(session, now+timedelta(minutes=2))
                assert second.polls[0].records == [] and second.events == []
                third = await load_event_inputs(session, now+timedelta(minutes=2))
                assert third.polls[0].poll_id == second.polls[0].poll_id
                assert third.latest_attempts['CME']['error'] is None
                assert third.latest_attempts['CME']['received_at'] == (now+timedelta(minutes=1)).isoformat()
        finally:
            await dispose_engine()
    asyncio.run(check())


def test_enlil_profiles_return_latest_successful_version(database, monkeypatch, object_store):
    _, _, environment = database
    migrate(environment)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    from clio.db.models.file_observation import FileObservation
    from clio.ingestion.utils.archive import store_receipt
    from common.db.session import dispose_engine
    from clio.db.session import get_db_session
    from clio.routers.solar_wind_events import load_event_inputs
    now = datetime.now(UTC).replace(microsecond=0)

    async def check():
        try:
            async with get_db_session() as session:
                for minute, speed in ((0, 500), (2, 900)):
                    metadata = dict(source_url='https://noaa-wsa-enlil-pds.s3.amazonaws.com/run/evo.l1.nc',
                        source_key='run/evo.l1.nc', source_modified_at=now-timedelta(minutes=1), model_created_at=now-timedelta(hours=1),
                        payload=dict(model_version='2.9', target='L1', density_kind='proton_equivalent',
                            points=[dict(valid_time=(now+timedelta(hours=1)).isoformat(), speed_km_s=speed, density_cm3=5)]))
                    session.add(FileObservation(**store_receipt('enlil', 'noaa.enlil', str(minute),
                        received_at=now+timedelta(minutes=minute), raw=str(speed).encode(), metadata=metadata)))
                await session.commit()
            async with get_db_session() as session:
                latest = await load_event_inputs(session, now+timedelta(minutes=3))
                assert latest.events == []
                assert latest.enlil_profile_status['state'] == 'available'
                assert latest.enlil_profiles[0].points[0].speed_km_s == 900
                assert latest.enlil_profile_status['latest_attempt']['error'] is None
                assert latest.enlil_profile_status['latest_attempt']['received_at'] == (now+timedelta(minutes=2)).isoformat()
                assert latest.enlil_profile_status['linked_to_donki'] is False
        finally:
            await dispose_engine()
    asyncio.run(check())


def test_closed_hourly_speed_history_includes_late_receipts_and_prefers_native_data(database, monkeypatch, object_store):
    _, _, environment = database
    migrate(environment)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    from clio.db.models.measurement import Measurement
    from common.db.session import dispose_engine
    from clio.db.session import get_db_session
    from clio.routers.solar_wind_events import hourly_speeds
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
