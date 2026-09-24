"""Exercise real migrations and monitoring persistence in a disposable Docker DB.

PYTHONPATH=apps/api:apps/clio/src:packages/common/src:packages/clio/src:apps/api/.venv/lib/python3.12/site-packages .venv/bin/python apps/api/tests/integration/verify_project_monitoring.py
No existing database or container is touched. The temporary container is stopped
in finally, including after a failed assertion.
"""
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

import pandas as pd
import psycopg
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.db.models.monitoring import MonitorState, TrafficMetric
from app.services import edge_traffic, project_monitoring
from argus_clio.services.observations.normalized import _upsert_measurements
from argus_clio.db.models import MeasurementReceipt

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
    clio = create_async_engine(dsn, connect_args={'options': '-csearch_path=clio,pg_catalog,pg_temp'})
    api_migrations = modules(ROOT/'apps/api/alembic/versions')
    clio_migrations = modules(ROOT/'apps/clio/src/argus_clio/migrations/versions')
    try:
        for name, engine, migrations in [('api', api, api_migrations), ('clio', clio, clio_migrations)]:
            async with engine.begin() as connection:
                await connection.execute(text(f'CREATE SCHEMA {name}'))
                await connection.run_sync(lambda c: migrate(c, migrations, 'upgrade'))
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
            with patch.dict(os.environ, {'MONITORING_TRAFFIC_LOG': str(log)}), patch.object(edge_traffic, 'get_session_factory', return_value=factory):
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
        clio_factory = async_sessionmaker(clio, expire_on_commit=False)
        async with clio_factory() as db:
            fresh = pd.DataFrame([{'metric': 'dst', 'value': -10, 'observed_at': now}])
            await _upsert_measurements(db, fresh)
            await db.commit()
            receipt = await db.get(MeasurementReceipt, 'dst')
            received = receipt.received_at
            from argus_clio.services.collection.monitoring import monitoring_status
            status = await monitoring_status(db)
            dst = next(m for m in status['measurements'] if m['metric'] == 'dst')
            assert dst['latest_observation_at'] == now and dst['received_at'] == received
            assert dst['status'] == 'fresh'
            assert status['sources'] and status['status'] == 'degraded'
            old = fresh.copy()
            old['observed_at'] = now-timedelta(days=1)
            await _upsert_measurements(db, old)
            await db.commit()
            await db.refresh(receipt)
            assert receipt.received_at == received and receipt.latest_observation_at == now
            await _upsert_measurements(db, fresh, track_receipt=False)
            await db.commit()
            await db.refresh(receipt)
            assert receipt.received_at == received
            uncommitted = fresh.copy()
            uncommitted['metric'] = 'v'
            await _upsert_measurements(db, uncommitted)
            await db.rollback()
            assert await db.get(MeasurementReceipt, 'v') is None
        for engine, migrations in [(clio, clio_migrations), (api, api_migrations)]:
            async with engine.begin() as connection:
                await connection.run_sync(lambda c: migrate(c, migrations, 'downgrade'))
                await connection.run_sync(lambda c: migrate(c, migrations, 'upgrade'))
        print('Monitoring integration passed: migration roundtrip, permissions, concurrent ingestion, durable cursors, rotation, snapshots, atomic receipts and older backfills.')
    finally:
        await api.dispose()
        await clio.dispose()


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
