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

from clio.db.models.measurement import Measurement
from clio.ingestion.conversion import wind_measurements
from clio.observations.measurements import store_native
from clio.observations.solar_wind import aggregated_history as history
from clio.observations.measurements import load_measurements

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
    scripts = ScriptDirectory(str(Path(__file__).resolve().parents[2] / 'src/clio/db/migrations'))
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
        migrate(conn, '20261003_unified_measurement', '20260929_gong_snapshot')
        tables = set(sa.inspect(conn).get_table_names(schema='clio'))
        assert not tables.intersection({'measurement_receipt', 'solar_wind_observation', 'geomagnetic_observation',
                                       'solar_wind_aggregate', 'solar_wind_retired_hour', 'solar_wind_aggregate_pending'})
        rows = {r.metric: r for r in conn.execute(sa.text('SELECT * FROM measurement'))}
        assert rows['bz'].value == -4
        assert rows['bx'].value is None and rows['bx'].quality == 'missing'
        assert rows['kp'].value == 3.33 and rows['ap'].value == 18
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
        engine = create_async_engine(database_url, connect_args={'options': '-csearch_path=clio,public,pg_catalog,pg_temp'})
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
                assert point['count'] == 3 and 'source_changes' not in point
                assert series['coverage']['expected_slots'] == 10
                assert series['coverage']['invalid_slots'] == 2 and series['coverage']['missing_slots'] == 5
                assert series['coverage']['percent'] == 30
                frame = await load_measurements(session, NOW)
                assert frame.loc[frame.metric == 'bz', 'value'].tolist() == [-8, 2, -2]
                session.add_all([Measurement(metric=metric, value=2, observed_at=NOW,
                    received_at=NOW+timedelta(minutes=5), quality='unverified', source_product=source)
                    for metric, source in [('kp', 'swpc.kp'), ('ap', 'swpc.kp'), ('dst', 'swpc.dst')]])
                await session.commit()
                from clio.routers.measurements import measurements
                result = await measurements(start=NOW, end=NOW+timedelta(hours=4), metrics='kp,ap,dst',
                    aggregation='raw', min_value=None, max_value=None, exclude_value=None, session=session)
                available = {row['metric']: row['available_at'] for row in result.data['items']}
                assert available == {'kp': NOW+timedelta(hours=3), 'ap': NOW+timedelta(hours=3),
                                     'dst': NOW+timedelta(hours=1)}
                payload[0]['values']['bz'] = -10
                await store_native(session, wind_measurements('mag', payload))
                await session.commit()
                receipt = await session.scalar(sa.select(Measurement.received_at).where(
                    Measurement.metric == 'bz', Measurement.observed_at == NOW))
                assert receipt == NOW+timedelta(hours=2)
        finally:
            await engine.dispose()
    asyncio.run(verify())


def test_backfill_normalize_and_cleanup_lifecycle(database_url, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import pandas as pd

    from clio.commands import cleanup
    from clio.config import load_observation_config
    from clio.db.models.normalized_observation import NormalizedObservation
    from clio.ingestion import numeric_backfill as backfill
    from clio.ingestion import normalize as normalized
    from clio.observations.schema import REQUIRED_METRICS

    config = load_observation_config().model_copy(deep=True)
    config.sdo_images.enabled = False
    config.observations = {m: config.observations[m] for m in REQUIRED_METRICS}
    for policy in config.observations.values():
        policy.backfill_days = 2
        policy.sources.historical = ['omni.hourly']
    now = NOW + timedelta(days=100)
    start = now - timedelta(days=2)
    archived = pd.DataFrame([dict(metric=metric, observed_at=start+timedelta(hours=h), value=2.)
                             for h in range(48) for metric in REQUIRED_METRICS])
    provider = Mock()
    provider.fetch.return_value = archived
    monkeypatch.setattr(backfill, 'historical_adapters', lambda: {'omni.hourly': provider})
    monkeypatch.setattr(normalized, 'load_observation_config', lambda: config)
    monkeypatch.setattr(cleanup, 'load_observation_config', lambda: config)

    async def verify():
        engine = create_async_engine(database_url)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        monkeypatch.setattr(cleanup, 'get_db_session', factory)
        try:
            async with engine.begin() as conn:
                for model in (Measurement, NormalizedObservation):
                    await conn.run_sync(model.__table__.create)
            existing = start + timedelta(hours=3)
            async with factory() as session:
                session.add(NormalizedObservation(observed_at=existing, **{m: 7. for m in REQUIRED_METRICS}))
                await session.commit()
                report = await backfill.backfill_numeric(session, config, list(REQUIRED_METRICS), now=now)
                assert report['status'] == 'complete'
                assert report['normalized_hours'] == 48
                assert await session.scalar(sa.select(sa.func.count()).select_from(NormalizedObservation)) == 48
                assert await session.scalar(sa.select(NormalizedObservation.bz).where(
                    NormalizedObservation.observed_at == existing)) == 7.
                # Filling a missing hourly row must also work when all raw history is already stored.
                await session.execute(sa.delete(NormalizedObservation).where(NormalizedObservation.observed_at == start))
                await session.commit()
                await backfill.backfill_numeric(session, config, list(REQUIRED_METRICS), now=now)
                provider.fetch.assert_called_once()
                assert await session.scalar(sa.select(NormalizedObservation.bz).where(
                    NormalizedObservation.observed_at == start)) == 2.
                live_hour = now - timedelta(hours=1)
                session.add_all([Measurement(metric=m, value=5., observed_at=live_hour+timedelta(minutes=15),
                    source_product=config.observations[m].sources.live[0], received_at=now, quality='unverified')
                    for m in REQUIRED_METRICS])
                await session.commit()
                await normalized.refresh_normalized_observations(session, now)
                assert await session.scalar(sa.select(NormalizedObservation.bz).where(
                    NormalizedObservation.observed_at == live_hour)) == 5.
                assert await session.scalar(sa.select(NormalizedObservation.bz).where(
                    NormalizedObservation.observed_at == existing)) == 7.
                assert await session.scalar(sa.select(Measurement.value).where(
                    Measurement.metric == 'bz', Measurement.observed_at == live_hour)) == 2.
                boundary = now - timedelta(days=6)
                old = now - timedelta(days=7)
                ancient = now - timedelta(days=90)
                session.add_all([
                    Measurement(metric='bz', value=1., observed_at=old, source_product='swpc.rtsw_mag'),
                    Measurement(metric='bz', value=1., observed_at=boundary, source_product='swpc.rtsw_mag'),
                    Measurement(metric='bz', value=2., observed_at=ancient, source_product='omni.hourly'),
                    Measurement(metric='v', value=3., observed_at=ancient, source_product=None),
                    NormalizedObservation(observed_at=now-timedelta(days=61), **{m: 2. for m in REQUIRED_METRICS}),
                    NormalizedObservation(observed_at=now-timedelta(days=60), **{m: 2. for m in REQUIRED_METRICS}),
                ])
                await session.commit()
            report = await cleanup.run(SimpleNamespace(now=now))
            assert report == {'live_removed': 1, 'normalized_removed': 1}
            async with factory() as session:
                rows = (await session.scalars(sa.select(Measurement).where(Measurement.observed_at <= boundary))).all()
                assert {(r.metric, r.observed_at) for r in rows} == {('bz', boundary), ('bz', ancient), ('v', ancient)}
                assert await session.scalar(sa.select(NormalizedObservation.observed_at).order_by(
                    NormalizedObservation.observed_at).limit(1)) == now-timedelta(days=60)
        finally:
            await engine.dispose()
    asyncio.run(verify())
