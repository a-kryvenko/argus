"""Separate databases on an explicitly disposable PostgreSQL server.

Never reads project .env credentials. Creates/drops only UUID-named databases
and owners. Run serially; TEST_DATABASE_ADMIN_DSN must be administrative.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4
from datetime import UTC, datetime, timedelta

import pytest

if not os.getenv('TEST_DATABASE_ADMIN_DSN'):
    pytest.skip('TEST_DATABASE_ADMIN_DSN is required for isolated PostgreSQL tests', allow_module_level=True)

import psycopg
from psycopg import sql
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


provisioning = load(ROOT / 'scripts/provision-databases.py')
transfer_checks = load(ROOT / 'scripts/transfer-databases.py')


def database_environment(urls):
    settings = {}
    for domain, value in urls.items():
        url = make_url(value)
        for field, value in dict(HOST=url.host, PORT=str(url.port or 5432), NAME=url.database,
                                 USER=url.username, PASSWORD=url.password).items():
            settings[domain.upper() + '_DB_' + field] = value
    return settings


@pytest.fixture
def database(tmp_path):
    admin_dsn = os.environ['TEST_DATABASE_ADMIN_DSN']
    admin_url = make_url(admin_dsn)
    token = uuid4().hex[:16]
    targets = {d: admin_url.set(database=f'test_{d}_{token}', username=f'test_{d}_{token}', password=uuid4().hex + '@:/%+?#')
               for d in provisioning.DOMAINS}
    urls = {d: u.render_as_string(hide_password=False) for d, u in targets.items()}
    dsns = {d: admin_url.set(database=u.database).render_as_string(hide_password=False) for d, u in targets.items()}
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/project.yaml').write_text('project: {name: test}\n')
    (tmp_path / 'configs/models_registry.yaml').write_text('models: {}\n')
    environment = {**os.environ, **database_environment(urls),
                   'ARGUS_WORKDIR': str(tmp_path),
                   'PYTHONPATH': ':'.join(str(ROOT / path) for path in (
                       'apps/api', 'apps/clio/src', 'apps/prophet/src', 'apps/intelligence/src',
                       'packages/common/src', 'packages/clio/src', 'packages/forecast/src'))}
    try:
        provisioning.provision(admin_dsn, urls, apply=True)
        yield dsns, urls, environment
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as admin:
            for url in targets.values():
                admin.execute(sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)').format(sql.Identifier(url.database)))
                admin.execute(sql.SQL('DROP ROLE IF EXISTS {}').format(sql.Identifier(url.username)))


def migrate(environment):
    for command in ([sys.executable, '-c', 'from argus_clio.cli import main; main()', 'migrate', 'upgrade', 'head'],
                    [sys.executable, '-m', 'alembic', '-c', str(ROOT / 'apps/api/alembic.ini'), 'upgrade', 'head'],
                    [sys.executable, '-c', 'from argus_prophet.cli import main; main()', 'migrate', 'upgrade', 'head'],
                    [sys.executable, '-c', 'from argus_intelligence.cli import main; main()', 'migrate', 'upgrade', 'head']):
        result = subprocess.run(command, env=environment, cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def runtime(dsns, domain, urls):
    return psycopg.connect(urls[domain], autocommit=True)


def test_fresh_databases_repeat_migrations_and_reject_foreign_connections(database):
    dsns, urls, environment = database
    migrate(environment)
    provisioning.provision(os.environ['TEST_DATABASE_ADMIN_DSN'], urls, apply=True)
    migrate(environment)
    for domain in provisioning.DOMAINS:
        with runtime(dsns, domain, urls) as conn:
            assert conn.execute(f'SELECT count(*) FROM {domain}.alembic_version').fetchone()[0] == 1
            # The one service owner can also perform its migrations.
            conn.execute(f'CREATE TABLE {domain}.owner_test(id int)')
            conn.execute(f'DROP TABLE {domain}.owner_test')
        for other in provisioning.DOMAINS:
            if other == domain:
                continue
            foreign = make_url(urls[domain]).set(database=make_url(urls[other]).database)
            with pytest.raises(psycopg.OperationalError):
                psycopg.connect(foreign.render_as_string(hide_password=False))
    with runtime(dsns, 'clio', urls) as conn:
        conn.execute('SET search_path TO pg_catalog')
        conn.execute("""INSERT INTO clio.solar_wind_observation(kind, observed_at, spacecraft, active, received_at, "values", raw)
            VALUES ('mag', '2026-09-02 00:00:00+00', 'A', true, '2026-09-02 00:00:00+00', '{"bz": -5}', '{}')""")
        assert conn.execute("SELECT count(*) FROM clio.solar_wind_aggregate_pending WHERE hour='2026-09-02 00:00:00+00'").fetchone()[0] == 1


def test_provisioning_does_not_rotate_existing_passwords(database):
    _, urls, _ = database
    changed = dict(urls)
    changed['api'] = make_url(urls['api']).set(password='wrong-password').render_as_string(hide_password=False)
    with pytest.raises(psycopg.OperationalError):
        provisioning.provision(os.environ['TEST_DATABASE_ADMIN_DSN'], changed, apply=True)
    with psycopg.connect(urls['api']) as conn:
        assert conn.execute('SELECT 1').fetchone()[0] == 1


def test_snapshot_handles_column_named_like_table_alias(database):
    dsns, urls, _ = database
    with runtime(dsns, 'clio', urls) as conn:
        conn.execute('CREATE SCHEMA clio')
        conn.execute('CREATE TABLE clio.alembic_version(version_num text PRIMARY KEY)')
        conn.execute("INSERT INTO clio.alembic_version VALUES ('test')")
        conn.execute('CREATE TABLE clio.alias_collision(t double precision, payload text)')
        conn.execute("INSERT INTO clio.alias_collision VALUES (100.5, 'original')")
    before = transfer_checks.snapshot(make_url(urls['clio']), 'clio')['tables']['alias_collision']
    assert before['count'] == 1
    with runtime(dsns, 'clio', urls) as conn:
        # The fingerprint must cover the whole record, not just the colliding column.
        conn.execute("UPDATE clio.alias_collision SET payload='changed'")
    after = transfer_checks.snapshot(make_url(urls['clio']), 'clio')['tables']['alias_collision']
    assert after['count'] == 1 and after['sha256'] != before['sha256']


def pg_tool(tool, url, arguments, *, input=None):
    """Use matching server tools in CI; local runs need pg_dump/pg_restore."""
    import shutil
    url = make_url(url)
    container = os.getenv('TEST_POSTGRES_CONTAINER')
    if not container and not shutil.which(tool):
        pytest.skip('PostgreSQL client tools or TEST_POSTGRES_CONTAINER required for transfer test')
    command = ['docker', 'exec', '-i', '-e', 'PGPASSWORD', container] if container else []
    command += [tool, '--host', url.host, '--port', str(url.port or 5432),
                '--username', url.username, '--dbname', url.database, *arguments]
    result = subprocess.run(command, input=input, capture_output=True,
                            env={**os.environ, 'PGPASSWORD': url.password})
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout


def test_schema_dump_transfer_preserves_rows_sequences_and_triggers(database, tmp_path, monkeypatch):
    dsns, urls, environment = database
    source_name = 'test_shared_' + uuid4().hex
    admin_dsn = os.environ['TEST_DATABASE_ADMIN_DSN']
    source_dsn = make_url(admin_dsn).set(database=source_name).render_as_string(hide_password=False)
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(source_name)))
    try:
        # Model the current shared database at the actual migration heads.
        source_env = {**environment, **database_environment({d: source_dsn for d in provisioning.DOMAINS})}
        migrate(source_env)
        run_id, attempt_id = uuid4(), uuid4()
        with psycopg.connect(source_dsn) as conn:
            conn.execute("INSERT INTO api.dashboard_user(username,password_hash,active) VALUES ('existing','hash',true)")
            conn.execute("INSERT INTO clio.measurement(metric,value,observed_at) VALUES ('f10_7',120,now())")
            conn.execute("""INSERT INTO clio.normalized_observation
                (observed_at,bx,by,bz,v,n,t,kp,dst,ap,f10_7)
                VALUES ('2026-09-02 00:00:00+00',1,2,-3,400,5,100000,2,-10,7,120)""")
            conn.execute("""INSERT INTO clio.solar_wind_observation(kind, observed_at, spacecraft, active, received_at, "values", raw)
                VALUES ('mag','2026-09-02 00:00:00+00','A',true,now(),'{"bz":-5}','{}')""")
            conn.execute("""INSERT INTO prophet.forecast_run(id,scope,product,trigger,started_at,status,provenance)
                VALUES (%s,'test','all','manual',now(),'succeeded','{}')""", (run_id,))
            conn.execute("INSERT INTO intelligence.attempt(id,product,status) VALUES (%s,'dst','skipped')", (attempt_id,))
        state = tmp_path / 'transfer-state'
        state.mkdir()
        for key, value in database_environment(urls).items():
            monkeypatch.setenv(key, value)
        source_url = make_url(source_dsn)
        for key, value in dict(HOST=source_url.host, PORT=str(source_url.port or 5432), NAME=source_name,
                               USER=source_url.username, PASSWORD=source_url.password).items():
            monkeypatch.setenv('DB_' + key, value)
        with pytest.raises(ValueError, match='does not match the installed'):
            transfer_checks.run('preflight', state, system_identifier='wrong-server')
        transfer_checks.run('preflight', state)
        with psycopg.connect(source_dsn) as active_client:
            active_client.execute('SELECT 1')
            with pytest.raises(ValueError, match='client sessions'):
                transfer_checks.run('freeze', state)
        transfer_checks.run('freeze', state)
        with psycopg.connect(source_dsn) as conn:
            assert conn.execute('SHOW default_transaction_read_only').fetchone()[0] == 'on'
        for domain in provisioning.DOMAINS:
            transfer_checks.run('snapshot', state, domain=domain)
            dump = pg_tool('pg_dump', source_dsn, ['--schema', domain, '--format=custom', '--no-owner', '--no-privileges'])
            pg_tool('pg_restore', dsns[domain], ['--role', make_url(urls[domain]).username,
                    '--no-owner', '--no-privileges', '--exit-on-error', '--single-transaction'], input=dump)
            transfer_checks.run('verify', state, domain=domain)
            with psycopg.connect(source_dsn) as source, runtime(dsns, domain, urls) as target:
                tables = source.execute("SELECT tablename FROM pg_tables WHERE schemaname=%s ORDER BY tablename", (domain,)).fetchall()
                for (table,) in tables:
                    query = sql.SQL('SELECT * FROM {}.{}').format(sql.Identifier(domain), sql.Identifier(table))
                    assert source.execute(query).fetchall() == target.execute(query).fetchall(), (domain, table)
                assert target.execute("SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=%s AND c.relowner != (SELECT oid FROM pg_roles WHERE rolname=current_user)", (domain,)).fetchone()[0] == 0
        transfer_checks.run('preflight', state)  # Completed restores are resumable.
        migrate(environment)  # Restored markers are respected; no duplicate creation.
        with runtime(dsns, 'api', urls) as conn:
            assert conn.execute("INSERT INTO api.dashboard_user(username,password_hash,active) VALUES ('next','hash',true) RETURNING id").fetchone()[0] == 2
        with pytest.raises(ValueError, match='does not match'):
            transfer_checks.run('verify', state, domain='api')
        with runtime(dsns, 'clio', urls) as conn:
            conn.execute('SET search_path TO pg_catalog')
            conn.execute("""INSERT INTO clio.solar_wind_observation(kind, observed_at, spacecraft, active, received_at, "values", raw)
                VALUES ('mag','2026-09-03 00:00:00+00','A',true,now(),'{"bz":-4}','{}')""")
            assert conn.execute("SELECT count(*) FROM clio.solar_wind_aggregate_pending WHERE hour='2026-09-03 00:00:00+00'").fetchone()[0] == 1
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as admin:
            admin.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(source_name)))


def test_scheduler_retries_restarts_and_serializes_jobs(database, monkeypatch):
    dsns, urls, environment = database
    migrate(environment)
    for key, value in environment.items():
        if key.startswith('CLIO_DB_'):
            monkeypatch.setenv(key, value)
    from argus_clio.scheduler import execute, JobBusy, JOBS
    now = datetime(2026, 9, 12, 12, tzinfo=UTC)
    calls = []
    def fail():
        raise RuntimeError('provider unavailable')
    with pytest.raises(RuntimeError, match='provider unavailable'):
        execute('refresh', fail, scheduled=True, now=now)
    assert execute('refresh', lambda: calls.append(1), scheduled=True, now=now)
    assert not execute('refresh', lambda: calls.append(2), scheduled=True, now=now)
    assert calls == [1]
    with runtime(dsns, 'clio', urls) as conn:
        conn.execute('SELECT pg_advisory_lock(%s)', (JOBS['refresh'][1],))
        with pytest.raises(JobBusy):
            execute('refresh', lambda: calls.append(3), now=now)
    assert execute('refresh', lambda: calls.append(4), scheduled=True, now=now + timedelta(hours=1))
    assert calls == [1, 4]
