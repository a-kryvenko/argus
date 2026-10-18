"""Receipt-time replay, empty replacement and failed-poll fallback in PostgreSQL."""
import asyncio
import gzip
import hashlib
from datetime import UTC, datetime, timedelta

from tests.support.database import database, migrate


def test_donki_versions_are_causal_and_empty_responses_replace_old_events(database, monkeypatch):
    _, _, environment = database
    migrate(environment)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    from clio.db.models.donki_poll import DonkiPoll
    from clio.db.session import get_session_factory, dispose_engine
    from clio.routers.solar_wind_events import load_event_inputs
    now = datetime.now(UTC)
    raw = b'[{"activityID":"cme1","startTime":"2026-01-01T00:00Z"}]'

    async def check():
        try:
            async with get_session_factory()() as session:
                for minute, payload, error in ((0, [{'activityID': 'cme1', 'startTime': '2026-01-01T00:00Z'}], None),
                                               (1, [], None), (2, None, 'Timeout')):
                    content = raw if payload else b'[]'
                    session.add(DonkiPoll(endpoint='CME', received_at=now+timedelta(minutes=minute),
                        request_start=now-timedelta(days=30), request_end=now+timedelta(days=1),
                        source_url='https://ccmc.gsfc.nasa.gov/DONKI-API/get/CME',
                        payload=payload, error=error,
                        sha256=hashlib.sha256(content).hexdigest() if payload is not None else None,
                        raw_gzip=gzip.compress(content) if payload is not None else None))
                await session.commit()
            async with get_session_factory()() as session:
                assert not (await load_event_inputs(session, now-timedelta(seconds=1), now)).polls
                first = await load_event_inputs(session, now, now)
                assert first.polls[0].records == [{'activityID': 'cme1', 'startTime': '2026-01-01T00:00Z'}]
                assert first.events[0].event_id == 'cme1'
                second = await load_event_inputs(session, now+timedelta(minutes=1), now)
                assert second.polls[0].records == [] and second.events == []
                third = await load_event_inputs(session, now+timedelta(minutes=2), now)
                assert third.polls[0].poll_id == second.polls[0].poll_id
                assert third.latest_attempts['CME']['error'] == 'Timeout'
        finally:
            await dispose_engine()
    asyncio.run(check())


def test_enlil_profiles_replay_receipts_versions_and_failures(database, monkeypatch):
    _, _, environment = database
    migrate(environment)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    from clio.db.models.enlil_profile import EnlilProfileReceipt
    from clio.db.session import get_session_factory, dispose_engine
    from clio.routers.solar_wind_events import load_event_inputs
    now = datetime.now(UTC).replace(microsecond=0)

    async def check():
        try:
            async with get_session_factory()() as session:
                for minute, speed in ((0, 500), (2, 900)):
                    session.add(EnlilProfileReceipt(received_at=now+timedelta(minutes=minute),
                        source_url='https://noaa-wsa-enlil-pds.s3.amazonaws.com/run/evo.l1.nc', source_key='run/evo.l1.nc',
                        source_modified_at=now-timedelta(minutes=1), model_created_at=now-timedelta(hours=1),
                        sha256=str(minute)*64, payload=dict(model_version='2.9', target='L1', density_kind='proton_equivalent',
                            points=[dict(valid_time=(now+timedelta(hours=1)).isoformat(), speed_km_s=speed, density_cm3=5)])))
                session.add(EnlilProfileReceipt(received_at=now+timedelta(minutes=3),
                    source_url='https://noaa-wsa-enlil-pds.s3.amazonaws.com/', error='Timeout'))
                await session.commit()
            async with get_session_factory()() as session:
                assert not (await load_event_inputs(session, now-timedelta(seconds=1), now)).enlil_profiles
                first = await load_event_inputs(session, now, now)
                assert first.events == []
                assert first.enlil_profiles[0].points[0].speed_km_s == 500
                assert first.enlil_profile_status['state'] == 'available'
                latest = await load_event_inputs(session, now+timedelta(minutes=3), now)
                assert latest.enlil_profiles[0].points[0].speed_km_s == 900
                assert latest.enlil_profile_status['latest_attempt']['error'] == 'Timeout'
                assert latest.enlil_profile_status['linked_to_donki'] is False
        finally:
            await dispose_engine()
    asyncio.run(check())


def test_closed_hourly_speed_history_prefers_covered_native_data_and_gates_receipts(database, monkeypatch):
    _, _, environment = database
    migrate(environment)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    from clio.db.models.measurement import Measurement
    from clio.db.session import get_session_factory, dispose_engine
    from clio.routers.solar_wind_events import hourly_speeds
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    first = now-timedelta(days=27)
    second = first+timedelta(hours=1)

    async def check():
        try:
            async with get_session_factory()() as session:
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
            async with get_session_factory()() as session:
                result = await hourly_speeds(session, now)
                assert len(result) == 2
                assert result[0].value == 600 and result[0].sample_count == 30
                assert result[1].value == 400 and result[1].source_product == 'omni.hourly'
                assert all(r.available_at <= now and r.interval_end <= now for r in result)
        finally:
            await dispose_engine()
    asyncio.run(check())
