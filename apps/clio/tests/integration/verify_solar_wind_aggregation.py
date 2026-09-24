"""Explicit PostgreSQL integration check, isolated schema, no provider requests."""
import asyncio
import importlib.util
from pathlib import Path
from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from uuid import uuid4
from dotenv import load_dotenv
from sqlalchemy import text, select, update
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from alembic.migration import MigrationContext
from alembic.operations import Operations
from argus_clio.db.models import SolarWindObservation, SolarWindAggregate, SolarWindAggregatePending
from argus_clio.services.solar_wind import aggregation as service
from argus_clio.services.solar_wind.audit import audit



def get_database_url():
    import os
    from sqlalchemy.engine import make_url
    value = os.getenv('TEST_DATABASE_ADMIN_DSN')
    if not value:
        raise RuntimeError('Set TEST_DATABASE_ADMIN_DSN to an isolated test PostgreSQL server')
    return make_url(value).set(drivername='postgresql+psycopg')


async def main():
    root = Path(__file__).resolve().parents[4]
    schema = 'aggregation_test_'+uuid4().hex
    admin = create_async_engine(get_database_url())
    db = create_async_engine(get_database_url(), connect_args={'options': f'-csearch_path={schema}'}, execution_options={'schema_translate_map': {'api': schema, 'clio': schema}})
    factory = async_sessionmaker(db, expire_on_commit=False)
    spec = importlib.util.spec_from_file_location('migration', root/'apps/clio/src/argus_clio/migrations/versions/20260908_0007_solar_wind_aggregates.py')
    migration = importlib.util.module_from_spec(spec); spec.loader.exec_module(migration)
    def migrate(connection, fn):
        with Operations.context(MigrationContext.configure(connection)): fn()
    created = False
    start = datetime(2026, 9, 1, tzinfo=UTC)
    try:
        async with admin.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA {schema}')); created = True
        async with db.begin() as conn:
            await conn.run_sync(SolarWindObservation.__table__.create)
            await conn.run_sync(lambda c: migrate(c, migration.upgrade))
        async with factory() as session:
            session.add(SolarWindObservation(kind='mag', observed_at=start, spacecraft='A',
                active=True, received_at=start, values={'bz': -5}, raw={}))
            await session.commit()
        with patch.object(service, 'get_session_factory', return_value=factory):
            assert await service.aggregate_pending() == 1
            assert await service.aggregate_pending() == 0
            async with factory() as session:
                rows = list((await session.execute(select(SolarWindAggregate))).scalars())
                assert len(rows) == 13
                first = next(r for r in rows if r.resolution_seconds == 3600)
                assert first.statistics['metrics']['bz']['min'] == -5
                old_time = first.calculated_at
                await session.execute(update(SolarWindObservation).values(values={'bz': -10}, raw={'revised': True}))
                await session.rollback()
            assert await service.aggregate_pending() == 0, 'rolled back raw update must not enqueue'
            async with factory() as session:
                await session.execute(update(SolarWindObservation).values(values={'bz': -10}, raw={'revised': True}))
                await session.commit()
            assert await service.aggregate_pending() == 1
            async with factory() as session:
                result = await session.get(SolarWindAggregate, ('mag', 3600, start))
                assert result.statistics['metrics']['bz']['min'] == -10
                assert result.calculated_at > old_time
                session.add(SolarWindAggregatePending(kind='mag', hour=start))
                await session.commit()
                revised_time = result.calculated_at
            assert await service.aggregate_pending() == 1
            async with factory() as session:
                result = await session.get(SolarWindAggregate, ('mag', 3600, start))
                assert result.calculated_at == revised_time, 'identical rebuild is a no-op'
                assert len(list((await session.execute(select(SolarWindAggregate))).scalars())) == 13
            # Simulate a worker failure: queue and aggregate writes must roll back together.
            async with factory() as session:
                session.add(SolarWindAggregatePending(kind='mag', hour=start)); await session.commit()
            with patch.object(service, 'summarize', side_effect=RuntimeError('interrupted')):
                try: await service.aggregate_pending()
                except RuntimeError: pass
                else: raise AssertionError('expected failure')
            assert await service.aggregate_pending() == 1
            # The first closed 5m window is written while the hour remains queued.
            # Repeated runs do not spin on that queue entry or produce partial rows.
            current = start+timedelta(days=1)
            async with factory() as session:
                session.add(SolarWindObservation(kind='plasma', observed_at=current, spacecraft='A',
                    active=True, received_at=current, values={'v': 400}, raw={}))
                await session.commit()
            assert await service.aggregate_pending(now=current+timedelta(minutes=7)) == 1
            async with factory() as session:
                rows = list((await session.execute(select(SolarWindAggregate).where(SolarWindAggregate.kind == 'plasma'))).scalars())
                assert len(rows) == 1 and rows[0].resolution_seconds == 300
                assert rows[0].statistics['window_complete']
                assert await session.get(SolarWindAggregatePending, ('plasma', current)) is not None
                saved_time = rows[0].calculated_at
            assert await service.aggregate_pending(now=current+timedelta(minutes=7)) == 1
            async with factory() as session:
                assert (await session.get(SolarWindAggregate, ('plasma', 300, current))).calculated_at == saved_time
            # No additional raw ingestion is needed for eventual hour completion.
            assert await service.aggregate_pending(now=current+timedelta(hours=1)) == 1
            assert await service.aggregate_pending(now=current+timedelta(hours=1)) == 0
            async with factory() as session:
                assert await session.get(SolarWindAggregate, ('plasma', 3600, current)) is not None
                assert await session.get(SolarWindAggregatePending, ('plasma', current)) is None
        async with db.begin() as conn:
            # Upgrade from the old worker: one-time requeue of partial windows.
            upgrade_spec = importlib.util.spec_from_file_location('closed_migration', root/'apps/clio/src/argus_clio/migrations/versions/20260908_0008_closed_aggregate_windows.py')
            closed_migration = importlib.util.module_from_spec(upgrade_spec); upgrade_spec.loader.exec_module(closed_migration)
            await conn.execute(text("UPDATE solar_wind_aggregate SET statistics = jsonb_set(statistics, '{window_complete}', 'false') WHERE kind = 'plasma'"))
            await conn.run_sync(lambda c: migrate(c, closed_migration.upgrade))
        with patch.object(service, 'get_session_factory', return_value=factory):
            assert await service.aggregate_pending() == 1
        # Audit the same retained raw data without modifying aggregates or queue.
        async with factory() as session:
            report = await audit(session, start, start+timedelta(days=2), start+timedelta(days=100))
            assert report['status'] == 'ok' and report['counts']['matched_buckets'] == 26
            assert report['storage']['older_rows'] == 2
            assert report['storage']['older_payload_bytes'] > 0
            assert report['storage']['estimated_older_allocation_bytes'] == report['storage']['allocated_bytes']
            assert (await session.execute(text('SHOW transaction_read_only'))).scalar_one() == 'on'
            assert (await session.execute(text('SHOW transaction_isolation'))).scalar_one() == 'repeatable read'
        async with factory() as session:
            await session.execute(text("DELETE FROM solar_wind_aggregate WHERE kind='mag' AND resolution_seconds=300 AND bucket_start=:start"), {'start': start})
            await session.execute(text("UPDATE solar_wind_aggregate SET statistics=jsonb_set(statistics, '{metrics,bz,mean}', '999') WHERE kind='mag' AND resolution_seconds=3600"))
            session.add(SolarWindAggregatePending(kind='mag', hour=start))
            await session.commit()
        async with factory() as session:
            report = await audit(session, start, start+timedelta(days=2), start+timedelta(days=100), detail_limit=2)
            assert report['status'] == 'issues_found'
            assert report['counts']['missing_buckets'] == 1
            assert report['counts']['mismatched_buckets'] == 1
            assert report['counts']['pending_source_hours'] == 1
            assert report['issue_count'] == 3 and report['details_truncated']
        async with factory() as session:
            # Removing raw data in this disposable schema must make the audit
            # report unverifiable, not certify an empty aggregate as correct.
            await session.execute(text("DELETE FROM solar_wind_observation WHERE kind='plasma'"))
            await session.commit()
        async with factory() as session:
            report = await audit(session, start, start+timedelta(days=2), start+timedelta(days=100))
            assert report['counts']['unverifiable_buckets'] == 13
            assert any('metrics.bz.mean' in issue.get('fields', []) for issue in report['issues'])
        async with db.begin() as conn:
            assert (await conn.execute(text("SELECT count(*) FROM solar_wind_aggregate WHERE statistics->>'window_complete' = 'false'"))).scalar_one() == 0
            await conn.run_sync(lambda c: migrate(c, closed_migration.downgrade))
            await conn.run_sync(lambda c: migrate(c, migration.downgrade))
        print('PASS: migration, durable queue, 5m/1h values, revisions, rollback, idempotency, recovery, downgrade')
    finally:
        await db.dispose()
        if created:
            async with admin.begin() as conn: await conn.execute(text(f'DROP SCHEMA {schema} CASCADE'))
        await admin.dispose()


if __name__ == '__main__': asyncio.run(main())
