"""Exercise monitoring persistence in a disposable Docker database."""
import asyncio
import importlib.util
import json
import os
import subprocess
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import psycopg
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from api.db.models.monitoring import MonitorState, TrafficMetric
from api.services.monitoring import edge_traffic
from api.services.monitoring import project as project_monitoring

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
        print('Monitoring integration passed: migration roundtrip, permissions, concurrent ingestion, durable cursors, rotation, snapshots.')
    finally:
        await api.dispose()


def main():
    name = 'argus-monitoring-test-' + uuid4().hex[:10]
    subprocess.run(['docker', 'run', '--rm', '-d', '--name', name,
        '-p', '127.0.0.1::5432', '-e', 'POSTGRES_PASSWORD=monitoring-test', 'postgres:17-alpine'], check=True, capture_output=True)
    try:
        binding = subprocess.check_output(['docker', 'port', name, '5432/tcp'], text=True).strip()
        port = binding.rsplit(':', 1)[1]
        dsn = f'postgresql://postgres:monitoring-test@127.0.0.1:{port}/postgres'
        for _ in range(30):
            try:
                with psycopg.connect(dsn, connect_timeout=1):
                    break
            except psycopg.OperationalError:
                time.sleep(.3)
        asyncio.run(verify(dsn.replace('postgresql://', 'postgresql+psycopg://')))
    finally:
        subprocess.run(['docker', 'stop', name], check=True, capture_output=True)


if __name__ == '__main__':
    main()
