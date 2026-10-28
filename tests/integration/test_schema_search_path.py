"""Runtime works before and after moving service tables to public."""
import subprocess

from psycopg import sql

from tests.support.database import ROOT, database, migrate, runtime


PROBE = '''
import asyncio
import importlib
from datetime import UTC, datetime
from sqlalchemy import delete, select, text
from common.db.session import dispose_engine
service = __import__('sys').argv[1]
base = importlib.import_module(service + '.db.base').Base
assert base.metadata.schema is None
get_db_session = importlib.import_module(service + '.db.session').get_db_session
async def verify():
    try:
        async with get_db_session() as session:
            await session.execute(text("INSERT INTO schema_path_probe VALUES (1, 'before')"))
            await session.execute(text("UPDATE schema_path_probe SET value='after' WHERE id=1"))
            assert await session.scalar(text('SELECT value FROM schema_path_probe WHERE id=1')) == 'after'
            await session.execute(text('DELETE FROM schema_path_probe WHERE id=1'))
            if service == 'api':
                from api.db.models.dashboard import User
                row = User(username='schema-probe', password_hash='test-only')
            elif service == 'clio':
                from clio.db.models.measurement import Measurement
                row = Measurement(metric='schema-probe', observed_at=datetime.now(UTC), value=1)
            else:
                table = 'forecast_release' if service == 'prophet' else 'drag_profile'
                await session.execute(text('SELECT * FROM ' + table + ' LIMIT 1'))
                row = None
            if row is not None:
                session.add(row)
                await session.flush()
                assert await session.scalar(select(type(row)).where(type(row).id == row.id)) is row
                await session.delete(row)
            await session.commit()
    finally:
        await dispose_engine()
asyncio.run(verify())
'''


AUTOGENERATE = '''
import sys
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.environment import EnvironmentContext
from alembic.script import ScriptDirectory
import importlib
from pathlib import Path
service = sys.argv[1]
scripts = ScriptDirectory(str(Path(importlib.import_module(service).__file__).parent / 'db/migrations'))
checked = []
def compare(revisions, context):
    metadata = importlib.import_module(service + '.db.base').Base.metadata
    assert metadata.schema is None
    diffs = compare_metadata(context, metadata)
    names = {table.name for table in metadata.tables.values()} | {'alembic_version'}
    assert not [diff for diff in diffs if isinstance(diff, tuple) and diff[0] in ('add_table', 'remove_table')
                and diff[1].name in names], diffs
    checked.append(True)
    return []
with EnvironmentContext(Config(), scripts, fn=compare):
    scripts.run_env()
assert checked
'''


def run_service(service, code, environment):
    return subprocess.run([str(ROOT / 'apps' / service / '.venv/bin/python'), '-c', code, service],
                          env={**environment, 'PYTHONPATH': ''}, cwd=ROOT, capture_output=True, text=True)


def test_runtime_resolves_both_layouts_and_metadata_does_not_recreate_tables(database):
    dsns, urls, environment = database
    migrate(environment)
    for service in ('api', 'clio', 'prophet', 'intelligence'):
        with runtime(dsns, service, urls) as connection:
            if service in ('clio', 'prophet'):
                assert connection.execute('SELECT to_regclass(%s)', (service + '.scheduled_job',)).fetchone()[0] is None
            connection.execute(sql.SQL('CREATE TABLE {}.schema_path_probe(id int PRIMARY KEY, value text)').format(
                sql.Identifier(service)))
        result = run_service(service, PROBE, environment)
        assert result.returncode == 0, result.stderr
        if service in ('api', 'clio'):
            result = run_service(service, AUTOGENERATE, environment)
            assert result.returncode == 0, result.stderr
        # Simulate the next release in disposable databases; this release moves no data.
        with runtime(dsns, service, urls) as connection:
            tables = connection.execute('''SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                WHERE n.nspname=%s AND c.relkind IN ('r','p')''', (service,)).fetchall()
            for (table,) in tables:
                connection.execute(sql.SQL('ALTER TABLE {}.{} SET SCHEMA public').format(
                    sql.Identifier(service), sql.Identifier(table)))
            connection.execute(sql.SQL('DROP SCHEMA {}').format(sql.Identifier(service)))
        result = run_service(service, PROBE, environment)
        assert result.returncode == 0, result.stderr
        status = subprocess.run([str(ROOT / 'apps' / service / '.venv/bin/python'),
                                 str(ROOT / 'scripts/deployment/migration_status.py'), service],
                                env=environment, cwd=ROOT, capture_output=True, text=True)
        assert status.returncode == 0 and status.stdout.strip() == 'current', status.stderr
