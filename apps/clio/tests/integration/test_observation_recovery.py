"""Run explicitly with the Clio Python environment; uses an isolated temporary schema.

apps/clio/.venv/bin/python apps/clio/tests/integration/test_observation_recovery.py
"""
import asyncio
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from clio.db.models.measurement import Measurement
from clio.ingestion import collect_numeric as live
from clio.providers import solar_wind as solar_wind, geomagnetic as geomagnetic



def get_database_url():
    import os
    from sqlalchemy.engine import make_url
    value = os.getenv('TEST_DATABASE_ADMIN_DSN')
    if not value:
        raise RuntimeError('Set TEST_DATABASE_ADMIN_DSN to an isolated test PostgreSQL server')
    return make_url(value).set(drivername='postgresql+psycopg')


async def verify():
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
            await connection.run_sync(Measurement.__table__.create)
        for source in ('mag', 'plasma', 'kp', 'dst'):
            solar = source in ('mag', 'plasma')
            provider = solar_wind if solar else geomagnetic
            product = f'swpc.rtsw_{source}' if solar else f'swpc.{source}'
            model = Measurement
            step = timedelta(seconds=60 if solar else {'kp': 10800, 'dst': 3600}[source])
            # More than one SQL batch for solar wind, eight hours of downtime.
            count = 601 if solar else 12
            start = datetime(2026, 9, 1, tzinfo=UTC)
            receipt = start + step * count
            records = []
            for index in range(count):
                at = start + step * index
                variables = list(solar_wind.FIELDS[source]) if solar else ['kp', 'ap'] if source == 'kp' else ['dst']
                records.append(dict(timestamp=at, **{key: 1. for key in variables}))
            source_filter = model.metric == (next(iter(solar_wind.FIELDS[source])) if solar else source)
            time_field = 'observed_at'

            async def stored():
                async with factory() as session:
                    return {getattr(row, time_field): row for row in
                            (await session.execute(select(model).where(source_filter))).scalars()}

            async def poll(payload, received_at=receipt):
                with patch.object(live, 'get_db_session', side_effect=factory), \
                     patch.object(provider, 'fetch_records', return_value=pd.DataFrame(payload)), \
                     patch('clio.ingestion.collect_numeric.datetime') as clock:
                    clock.now.return_value = received_at
                    metrics = list(solar_wind.FIELDS[source]) if solar else ['kp', 'ap'] if source == 'kp' else ['dst']
                    await live.collect_product(product, {metric: [product] for metric in metrics}, start, receipt)

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
            await poll(repeated, receipt+timedelta(hours=8))
            rows = await stored()
            assert len(rows) == count-1, 'Repeat must not create duplicates'
            assert all(row.received_at == receipt for row in rows.values())

            # Late publication repairs the remaining hole; revisions replace old values.
            revision = deepcopy(records[2])
            revision.update({key: 2. for key in variables})
            revised_at = receipt+timedelta(hours=9)
            await poll([records[3], revision], revised_at)
            rows = await stored()
            assert len(rows) == count
            revised = rows[start+step*2]
            assert revised.received_at == revised_at
            assert revised.value == 2
            print(f'PASS {source}: downtime recovery, internal gaps, retained history, no duplicates, stable receipts, late data and revisions')
    finally:
        await db.dispose()
        if created:
            async with admin.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA {schema} CASCADE'))
        await admin.dispose()


if __name__ == '__main__':
    asyncio.run(verify())


def test_database_contract():
    import os
    import pytest
    if not os.getenv('TEST_DATABASE_ADMIN_DSN'):
        pytest.skip('Set TEST_DATABASE_ADMIN_DSN to an isolated PostgreSQL server')
    asyncio.run(verify())
