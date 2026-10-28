"""Monitoring persistence against a disposable PostgreSQL database."""
import asyncio
import importlib.util
import json
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import text, select
from sqlalchemy.engine import make_url

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from api.db.models.monitoring import MonitorState, TrafficMetric
from api.services.monitoring import edge_traffic
from api.services.monitoring import project as project_monitoring
from tests.support.database import database

ROOT = Path(__file__).resolve().parents[4]


def modules(directory):
    remaining = []
    for file in sorted(directory.glob('*.py')):
        if file.name == '__init__.py':
            continue
        spec = importlib.util.spec_from_file_location(file.stem, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        remaining.append(module)
    ordered, revision = [], None
    while remaining:
        module = next(m for m in remaining if m.down_revision == revision)
        ordered.append(module)
        remaining.remove(module)
        revision = module.revision
    return ordered


def migrate(connection, migrations, direction):
    with Operations.context(MigrationContext.configure(connection)):
        for migration in migrations if direction == 'upgrade' else reversed(migrations):
            getattr(migration, direction)()


async def verify(dsn):
    api = create_async_engine(dsn, connect_args={'options': '-csearch_path=api,pg_catalog,pg_temp'})
    api_migrations = modules(ROOT/'apps/api/src/api/db/migrations/versions')
    try:
        async with api.begin() as connection:
            await connection.execute(text('CREATE SCHEMA api'))
            await connection.run_sync(lambda c: migrate(c, api_migrations, 'upgrade'))
        factory = async_sessionmaker(api, expire_on_commit=False)
        async with factory() as db:
            permissions = await db.scalar(text("SELECT permissions FROM dashboard_group WHERE name='admins'"))
            assert 'project_monitoring.read' in permissions and 'users.manage' in permissions
            assert await db.scalar(text("SELECT permissions FROM dashboard_group WHERE name='clients'")) == []
        now = datetime.now(UTC)
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/'traffic.jsonl'
            def record(channel='api'):
                return json.dumps({'channel': channel, 'status': 200, 'time': now.timestamp(), 'seconds': '0.025'})+'\n'
            log.write_text(record()+record('site')+record('monitor'))
            with patch.dict(os.environ, {'MONITORING_TRAFFIC_LOG': str(log)}), patch.object(edge_traffic, 'get_db_session', side_effect=factory):
                await asyncio.gather(edge_traffic.ingest(), edge_traffic.ingest())
                async with factory() as db:
                    assert await db.scalar(select(text('sum(count)')).select_from(TrafficMetric)) == 4
                    state = await db.get(MonitorState, 'traffic')
                    assert state.payload['caught_up'] and state.payload['last_event_at']
                # Simulate a later poll, preserving its durable cursor.
                async def age_cursor():
                    async with factory() as db:
                        await db.execute(text("UPDATE monitor_state SET checked_at=checked_at-interval '1 minute' WHERE name='traffic'"))
                        await db.commit()
                await age_cursor()
                await edge_traffic.ingest()
                async with factory() as db:
                    assert await db.scalar(select(text('sum(count)')).select_from(TrafficMetric)) == 4
                with log.open('a') as stream:
                    stream.write(record('site'))
                log.rename(Path(directory)/'traffic.jsonl.1')
                log.write_text(record())
                for _ in range(2):
                    await age_cursor()
                    await edge_traffic.ingest()
                async with factory() as db:
                    assert await db.scalar(select(text('sum(count)')).select_from(TrafficMetric)) == 8
                    await project_monitoring.save_state(db, 'project', {'status': 'degraded', 'when': now}, now)
                    await db.commit()
                    assert (await db.get(MonitorState, 'project')).payload['when']
        async with api.begin() as connection:
            await connection.run_sync(lambda c: migrate(c, api_migrations, 'downgrade'))
            await connection.run_sync(lambda c: migrate(c, api_migrations, 'upgrade'))
    finally:
        await api.dispose()


def test_monitoring_persistence(database):
    _, urls, _ = database
    dsn = make_url(urls['api']).set(drivername='postgresql+psycopg')
    asyncio.run(verify(dsn))
