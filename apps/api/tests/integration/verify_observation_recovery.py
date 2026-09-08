"""Run explicitly with the API Python environment; uses an isolated temporary schema.

PYTHONPATH=apps/api apps/api/.venv/bin/python apps/api/tests/integration/verify_observation_recovery.py
"""
import asyncio
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import SolarWindObservation, GeomagneticObservation, ObservationSourceStatus
from app.db.session import get_database_url
from app.services import solar_wind, geomagnetic


async def verify():
    root = Path(__file__).resolve().parents[4]
    load_dotenv(root / '.env')
    load_dotenv(root / '.env.local', override=True)
    schema = 'recovery_test_' + uuid4().hex
    admin = create_async_engine(get_database_url())
    db = create_async_engine(get_database_url(), connect_args={'options': f'-csearch_path={schema}'})
    factory = async_sessionmaker(db, expire_on_commit=False)
    created = False
    try:
        async with admin.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA {schema}'))
            created = True
        async with db.begin() as connection:
            for model in (SolarWindObservation, GeomagneticObservation, ObservationSourceStatus):
                await connection.run_sync(model.__table__.create)
        for source in ('mag', 'plasma', 'kp', 'dst'):
            solar = source in ('mag', 'plasma')
            service = solar_wind if solar else geomagnetic
            model = SolarWindObservation if solar else GeomagneticObservation
            step = timedelta(seconds=60 if solar else geomagnetic.INTERVAL_SECONDS[source])
            # More than one SQL batch for solar wind, eight hours of downtime.
            count = 601 if solar else 12
            start = datetime(2026, 9, 1, tzinfo=UTC)
            receipt = start + step * count
            records = []
            for index in range(count):
                at = start + step * index
                raw = {'sample': index, 'value': 1}
                if solar:
                    records.append(dict(kind=source, observed_at=at, spacecraft='test', active=True,
                        received_at=receipt, values={key: 1 for key in solar_wind.FIELDS[source]}, raw=raw))
                else:
                    records.append(dict(metric=source, interval_start=at, interval_end=at+step,
                        received_at=receipt, value=1, quality='unverified', raw=raw))
            source_filter = model.kind == source if solar else model.metric == source
            time_field = 'observed_at' if solar else 'interval_start'

            async def stored():
                async with factory() as session:
                    return {getattr(row, time_field): row for row in
                            (await session.execute(select(model).where(source_filter))).scalars()}

            async def poll(payload):
                with patch.object(service, 'get_session_factory', return_value=factory), \
                     patch.object(service, 'fetch_records', return_value=deepcopy(payload)):
                    await service.ingest_source(source)

            # Existing prefix plus a newer point: recovery must fill internal holes too.
            await poll(records[:2] + records[-1:])
            await poll(records[1:3] + records[4:])  # provider itself has no index 3
            rows = await stored()
            assert len(rows) == count-1, (source, 'missing recovery rows')
            assert start in rows, 'Older data outside provider window must survive'
            assert start+step*3 not in rows, 'Do not fabricate missing source data'
            assert all(start+step*i in rows for i in range(4, count))

            # A repeated response has a new receipt time but identical source content.
            repeated = deepcopy(records[1:3] + records[4:])
            for row in repeated:
                row['received_at'] += timedelta(hours=8)
            await poll(repeated)
            rows = await stored()
            assert len(rows) == count-1, 'Repeat must not create duplicates'
            assert all(row.received_at == receipt for row in rows.values())

            # Late publication repairs the remaining hole; revisions replace old values.
            revision = deepcopy(records[2])
            revision['raw']['value'] = 2
            revision['received_at'] += timedelta(hours=9)
            if solar:
                revision['values'] = {key: 2 for key in solar_wind.FIELDS[source]}
            else:
                revision['value'] = 2
            await poll([records[3], revision])
            rows = await stored()
            assert len(rows) == count
            revised = rows[start+step*2]
            assert revised.raw['value'] == 2 and revised.received_at == revision['received_at']
            assert (all(value == 2 for value in revised.values.values()) if solar else revised.value == 2)
            print(f'PASS {source}: downtime recovery, internal gaps, retained history, no duplicates, stable receipts, late data and revisions')
    finally:
        await db.dispose()
        if created:
            async with admin.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA {schema} CASCADE'))
        await admin.dispose()


if __name__ == '__main__':
    asyncio.run(verify())
