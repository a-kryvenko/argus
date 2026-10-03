"""Migration and canonical observation contracts on a disposable database."""
import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from clio.db.models import Measurement
from clio.observations.native import wind_measurements, store_native
from clio.domains.solar_wind.history import history
from clio.observations.store import load_measurements

pytestmark = pytest.mark.skipif(not os.getenv('TEST_DATABASE_ADMIN_DSN'), reason='Disposable PostgreSQL required')
NOW = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture
def database_url():
    import psycopg
    from psycopg import sql
    admin = sa.make_url(os.environ['TEST_DATABASE_ADMIN_DSN'])
    name = 'test_unified_' + uuid4().hex
    with psycopg.connect(admin.render_as_string(hide_password=False), autocommit=True) as connection:
        connection.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
    url = admin.set(database=name, drivername='postgresql+psycopg')
    try:
        yield url
    finally:
        with psycopg.connect(admin.render_as_string(hide_password=False), autocommit=True) as connection:
            connection.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))


def migrate(connection, target, start='base'):
    scripts = ScriptDirectory(str(Path(__file__).resolve().parents[2] / 'src/clio/migrations'))
    with Operations.context(MigrationContext.configure(connection)):
        for revision in reversed(list(scripts.iterate_revisions(target, start))):
            revision.module.upgrade()


def test_migration_transfers_selected_values_before_removing_tables(database_url):
    engine = sa.create_engine(database_url)
    with engine.begin() as conn:
        conn.execute(sa.text('CREATE SCHEMA clio'))
        conn.execute(sa.text('SET search_path TO clio'))
        migrate(conn, '20260929_gong_snapshot')
        seed = """
            INSERT INTO measurement(metric,value,observed_at,source_product) VALUES
                ('bz',99,:at,'swpc.propagated_magnetic'), ('kp',2,:at,'swpc.kp'), ('f10_7',150,:at,'gfz.f107');
            INSERT INTO measurement_receipt VALUES ('f10_7',:at,:at);
            INSERT INTO solar_wind_observation(kind,observed_at,spacecraft,active,received_at,"values",raw) VALUES
                ('mag',:at,'A',true,:at,'{"bz": -4,"bt": 6,"bx": null,"by": 1}', '{"overall_quality": 0}'),
                ('mag',:at,'B',false,:at,'{"bz": 99}', '{}');
            INSERT INTO geomagnetic_observation(metric,interval_start,interval_end,value,quality,received_at,raw) VALUES
                ('kp',:at,:at + interval '3h',3.33,'unverified',:at,'{"a_running": 18,"station_count": 8}'),
                ('dst',:at,:at + interval '1h',null,'missing',:at,'{}');
        """
        for statement in seed.split(';'):
            if statement.strip():
                conn.execute(sa.text(statement), {'at': NOW})
        migrate(conn, 'head', '20260929_gong_snapshot')
        tables = set(sa.inspect(conn).get_table_names(schema='clio'))
        assert not tables.intersection({'measurement_receipt', 'solar_wind_observation', 'geomagnetic_observation',
                                       'solar_wind_aggregate', 'solar_wind_retired_hour', 'solar_wind_aggregate_pending'})
        rows = {r.metric: r for r in conn.execute(sa.text('SELECT * FROM measurement'))}
        assert rows['bz'].value == -4 and rows['bz'].spacecraft == 'A'
        assert rows['bx'].value is None and rows['bx'].quality == 'missing'
        assert rows['kp'].value == 3.33 and rows['ap'].value == 18
        assert rows['kp'].interval_end == NOW+timedelta(hours=3)
        assert rows['dst'].value is None and rows['dst'].quality == 'missing'
        assert rows['f10_7'].received_at == NOW
    engine.dispose()


def test_statistics_and_receipts_use_only_canonical_measurements(database_url):
    engine = sa.create_engine(database_url)
    with engine.begin() as conn:
        conn.execute(sa.text('CREATE SCHEMA clio'))
        conn.execute(sa.text('SET search_path TO clio'))
        migrate(conn, 'head')
    engine.dispose()

    async def verify():
        engine = create_async_engine(database_url)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with factory() as session:
                payload = [dict(observed_at=NOW+timedelta(minutes=i), received_at=NOW+timedelta(hours=1),
                                spacecraft='A' if i < 2 else 'B', active=True,
                                values={'bz': value}, raw={'overall_quality': 1 if i == 2 else 0})
                           for i, value in enumerate([-8, 2, 99, None, -2])]
                await store_native(session, wind_measurements('mag', payload))
                await session.commit()
                for row in payload:
                    row['received_at'] += timedelta(hours=1)
                await store_native(session, wind_measurements('mag', payload))
                await session.commit()
                receipt = await session.scalar(sa.select(Measurement.received_at).where(Measurement.metric == 'bz').limit(1))
                assert receipt == NOW+timedelta(hours=1)
                data = await history(session, ['bz'], NOW, NOW+timedelta(minutes=10), 300, now=NOW+timedelta(hours=3))
                series = data['series']['bz']
                point = series['points'][0]
                assert point['value'] == pytest.approx(-8/3)
                assert point['min'] == -8 and point['max'] == 2 and point['negative_count'] == 2
                assert point['count'] == 3 and point['source_changes'] == 1
                assert series['coverage']['expected_slots'] == 10
                assert series['coverage']['invalid_slots'] == 2 and series['coverage']['missing_slots'] == 5
                assert series['coverage']['percent'] == 30
                frame = await load_measurements(session, NOW)
                assert frame.loc[frame.metric == 'bz', 'value'].tolist() == [-8, 2, -2]
                payload[0]['values']['bz'] = -10
                await store_native(session, wind_measurements('mag', payload))
                await session.commit()
                receipt = await session.scalar(sa.select(Measurement.received_at).where(
                    Measurement.metric == 'bz', Measurement.observed_at == NOW))
                assert receipt == NOW+timedelta(hours=2)
        finally:
            await engine.dispose()
    asyncio.run(verify())
