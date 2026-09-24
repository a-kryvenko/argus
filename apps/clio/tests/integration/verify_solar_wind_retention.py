"""Retention integration check. Deletes synthetic data only in a disposable schema."""
import asyncio
import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
from dotenv import load_dotenv
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from alembic.migration import MigrationContext
from alembic.operations import Operations
from argus_clio.db.models import SolarWindObservation, SolarWindAggregate, SolarWindAggregatePending, SolarWindRetiredHour
from argus_clio.services.solar_wind import retention
from argus_clio.services.solar_wind import aggregation

START = datetime(2026, 1, 1, tzinfo=UTC)
NOW = START+timedelta(days=100, minutes=30)



def get_database_url():
    import os
    from sqlalchemy.engine import make_url
    value = os.getenv('TEST_DATABASE_ADMIN_DSN')
    if not value:
        raise RuntimeError('Set TEST_DATABASE_ADMIN_DSN to an isolated test PostgreSQL server')
    return make_url(value).set(drivername='postgresql+psycopg')


async def main():
    root = Path(__file__).resolve().parents[4]
    schema = 'retention_test_'+uuid4().hex
    admin = create_async_engine(get_database_url())
    db = create_async_engine(get_database_url(), connect_args={'options': f'-csearch_path={schema}'}, execution_options={'schema_translate_map': {'api': schema, 'clio': schema}})
    factory = async_sessionmaker(db, expire_on_commit=False)
    created = False

    def migrate(conn, suffix, operation):
        file = next((root/'apps/clio/src/argus_clio/migrations/versions').glob(f'*_{suffix}.py'))
        spec = importlib.util.spec_from_file_location('migration', file)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with Operations.context(MigrationContext.configure(conn)):
            getattr(module, operation)()

    def raw(hour, spacecraft='A', active=True):
        return SolarWindObservation(kind='mag', observed_at=hour, received_at=hour,
                                   spacecraft=spacecraft, active=active, values={'bz': -5}, raw={})

    async def snapshot():
        async with factory() as session:
            return [(r.kind, r.resolution_seconds, r.bucket_start, r.calculated_at, r.statistics)
                    for r in (await session.execute(select(SolarWindAggregate).order_by(
                        SolarWindAggregate.kind, SolarWindAggregate.resolution_seconds, SolarWindAggregate.bucket_start))).scalars()]

    try:
        async with admin.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA {schema}')); created = True
        async with db.begin() as conn:
            await conn.run_sync(SolarWindObservation.__table__.create)
            await conn.run_sync(lambda c: migrate(c, '0007_solar_wind_aggregates', 'upgrade'))
            await conn.run_sync(lambda c: migrate(c, '0009_solar_wind_retention', 'upgrade'))
            await conn.run_sync(lambda c: migrate(c, '0009_solar_wind_retention', 'downgrade'))
            await conn.run_sync(lambda c: migrate(c, '0009_solar_wind_retention', 'upgrade'))
        boundary = START+timedelta(days=10)
        async with factory() as session:
            session.add_all([raw(START+timedelta(hours=i)) for i in range(5)])
            session.add(raw(START, 'B', False))
            session.add(raw(boundary))
            await session.commit()
        with patch.object(aggregation, 'get_session_factory', return_value=factory), patch.object(retention, 'get_session_factory', return_value=factory):
            assert await aggregation.aggregate_pending(now=NOW) == 6
            async with factory() as session:
                await session.execute(text("UPDATE solar_wind_observation SET values='{}', raw=jsonb_build_object('revised', true) WHERE observed_at=:hour"), {'hour': START+timedelta(hours=1)})
                await session.execute(text("UPDATE solar_wind_aggregate SET statistics=jsonb_set(statistics,'{metrics,bz,min}','-99') WHERE bucket_start=:hour AND resolution_seconds=3600"), {'hour': START+timedelta(hours=2)})
                await session.execute(text('DELETE FROM solar_wind_aggregate WHERE bucket_start=:hour AND resolution_seconds=3600'), {'hour': START+timedelta(hours=3)})
                await session.commit()
            before = await snapshot()
            report = await retention.cleanup(now=NOW)
            assert report['mode'] == 'report' and report['deleted_rows'] == 0
            assert report['examined_hours'] == 5 and report['eligible_rows'] == 3
            assert {e.get('reason') for e in report['hours'] if e['status']=='skipped'} == {'recalculation_pending', 'aggregate_mismatch', 'aggregate_missing'}
            async with factory() as session:
                assert len(list((await session.execute(select(SolarWindObservation))).scalars())) == 7
                assert not list((await session.execute(select(SolarWindRetiredHour))).scalars())
            # An active writer prevents apply; cleanup must not wait or delete.
            async with db.begin() as conn:
                await conn.execute(text('LOCK TABLE solar_wind_observation IN ROW EXCLUSIVE MODE'))
                busy = await retention.cleanup(apply=True, start=START+timedelta(hours=4), limit=1, now=NOW)
                assert busy['hours'][0]['reason'] == 'database_busy'
            # Force a database failure during delete; marker and rows roll back together.
            async with db.begin() as conn:
                await conn.execute(text("""
                    CREATE FUNCTION fail_retention_test() RETURNS trigger LANGUAGE plpgsql AS $$
                    BEGIN RAISE EXCEPTION 'synthetic delete failure'; END $$;
                    CREATE TRIGGER fail_retention_test AFTER DELETE ON solar_wind_observation
                    FOR EACH ROW EXECUTE FUNCTION fail_retention_test();
                """))
            try:
                await retention.cleanup(apply=True, start=START+timedelta(hours=4), limit=1, now=NOW)
            except DBAPIError:
                pass
            else:
                raise AssertionError('Expected synthetic delete failure')
            async with db.begin() as conn:
                await conn.execute(text('DROP TRIGGER fail_retention_test ON solar_wind_observation'))
                await conn.execute(text('DROP FUNCTION fail_retention_test()'))
            async with factory() as session:
                assert await session.get(SolarWindRetiredHour, ('mag', START+timedelta(hours=4))) is None
            assert await snapshot() == before
            applied = await retention.cleanup(apply=True, now=NOW)
            assert applied['deleted_rows'] == 3 and applied['skipped_hours'] == 3
            assert await snapshot() == before, 'Aggregate values/timestamps must not change'
            async with factory() as session:
                assert len(list((await session.execute(select(SolarWindRetiredHour))).scalars())) == 2
                assert await session.get(SolarWindObservation, ('mag', boundary, 'A')) is not None
                assert await session.get(SolarWindAggregatePending, ('mag', START)) is None
            repeat = await retention.cleanup(apply=True, now=NOW)
            assert repeat['deleted_rows'] == 0
            # Both replaying old observations and explicitly requeuing retired hours fail.
            for record in (raw(START), SolarWindAggregatePending(kind='mag', hour=START)):
                try:
                    async with factory() as session:
                        session.add(record); await session.commit()
                except DBAPIError as exc:
                    assert 'retired' in str(exc)
                else:
                    raise AssertionError('Retired hour accepted new raw data or a rebuild')
            # Ordinary deletions still enqueue their hour.
            async with factory() as session:
                await session.execute(text('DELETE FROM solar_wind_aggregate_pending'))
                await session.execute(text('DELETE FROM solar_wind_observation WHERE observed_at=:hour'), {'hour': START+timedelta(hours=1)})
                await session.commit()
                assert await session.get(SolarWindAggregatePending, ('mag', START+timedelta(hours=1))) is not None
            try:
                async with db.begin() as conn:
                    await conn.run_sync(lambda c: migrate(c, '0009_solar_wind_retention', 'downgrade'))
            except DBAPIError:
                pass
            else:
                raise AssertionError('Protection removed after destructive cleanup')
        print('PASS: report, apply, boundaries, missing/mismatched/pending skips, writer contention, rollback, preserved aggregates, replay/requeue protection, idempotency, downgrade guard')
    finally:
        await db.dispose()
        if created:
            async with admin.begin() as conn: await conn.execute(text(f'DROP SCHEMA {schema} CASCADE'))
        await admin.dispose()


if __name__ == '__main__':
    asyncio.run(main())
