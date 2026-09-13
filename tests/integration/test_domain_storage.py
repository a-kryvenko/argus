"""Real PostgreSQL tests. TEST_DATABASE_ADMIN_DSN must target a disposable server.

Creates/drops disposable databases and provisions the four reserved Argus roles.
Never reads project .env credentials. Run this serially on its own test server.
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
from psycopg.conninfo import make_conninfo
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from alembic.migration import MigrationContext
from alembic.operations import Operations

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bootstrap = load(ROOT / 'scripts/bootstrap-domain-db.py')


@pytest.fixture
def database(tmp_path):
    dsn = os.getenv('TEST_DATABASE_ADMIN_DSN')
    if not dsn:
        pytest.skip('TEST_DATABASE_ADMIN_DSN is required for isolated PostgreSQL tests')
    name = 'argus_test_' + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
    url = make_url(dsn).set(database=name)
    target = url.render_as_string(hide_password=False)
    passwords = {key: uuid4().hex for key in ('API_DB_PASSWORD', 'API_MIGRATION_PASSWORD',
                                             'CLIO_DB_PASSWORD', 'CLIO_MIGRATION_PASSWORD',
                                             'PROPHET_DB_PASSWORD', 'PROPHET_MIGRATION_PASSWORD')}
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/project.yaml').write_text('project: {name: test}\n')
    (tmp_path / 'configs/models_registry.yaml').write_text('models: {}\n')
    environment = {**os.environ, **passwords, 'DB_NAME': name, 'DB_HOST': url.host or 'localhost',
                   'DB_PORT': str(url.port or 5432), 'ARGUS_WORKDIR': str(tmp_path),
                   'PYTHONPATH': ':'.join(str(ROOT / path) for path in (
                       'apps/api', 'apps/clio/src', 'apps/prophet/src', 'packages/common/src',
                       'packages/clio/src', 'packages/forecast/src'))}
    try:
        yield target, passwords, environment
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))


def migrate(environment):
    for command in ([sys.executable, '-c', 'from argus_clio.cli import main; main()', 'migrate', 'upgrade', 'head'],
                    [sys.executable, '-m', 'alembic', '-c', str(ROOT / 'apps/api/alembic.ini'), 'upgrade', 'head'],
                    [sys.executable, '-c', 'from argus_prophet.cli import main; main()', 'migrate', 'upgrade', 'head']):
        result = subprocess.run(command, env=environment, cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def legacy_database(dsn):
    engine = create_engine(make_url(dsn).set(drivername='postgresql+psycopg'))
    with engine.begin() as connection:
        files = sorted((ROOT / 'apps/clio/src/argus_clio/migrations/versions').glob('2026*.py'))
        files = [p for p in files if 'clio_jobs' not in p.name]
        files += [ROOT / 'apps/api/alembic/versions/20260911_0010_dashboard.py']
        with Operations.context(MigrationContext.configure(connection)):
            for path in files:
                load(path).upgrade()
        connection.execute(text("CREATE TABLE public.alembic_version (version_num varchar(32) PRIMARY KEY)"))
        connection.execute(text("INSERT INTO public.alembic_version VALUES ('20260911_0010')"))
        connection.execute(text("INSERT INTO measurement(metric, value, observed_at) VALUES ('f10_7', 120, '2026-09-01 00:00:00+00')"))
        connection.execute(text("INSERT INTO dashboard_user(username, password_hash, active) VALUES ('existing', 'hash', true)"))
        connection.execute(text("""INSERT INTO solar_wind_observation(kind, observed_at, spacecraft, active, received_at, "values", raw)
            VALUES ('mag', '2026-09-01 00:00:00+00', 'A', true, '2026-09-01 00:00:00+00', '{"bz": -5}', '{}')"""))
        connection.execute(text("""INSERT INTO solar_wind_aggregate(kind, resolution_seconds, bucket_start, version, calculated_at, statistics)
            VALUES ('mag', 3600, '2026-08-01 00:00:00+00', 1, '2026-09-01 00:00:00+00', '{"retained": true}')"""))
        connection.execute(text("""INSERT INTO solar_wind_retired_hour(kind, hour, retired_at, raw_rows)
            VALUES ('mag', '2026-08-01 00:00:00+00', '2026-09-01 00:00:00+00', 60)"""))
    engine.dispose()


def runtime(dsn, domain, passwords, migrator=False):
    role = f'argus_{domain}' + ('_migrator' if migrator else '')
    key = domain.upper() + ('_MIGRATION_PASSWORD' if migrator else '_DB_PASSWORD')
    return psycopg.connect(dsn, user=role, password=passwords[key], autocommit=True)


@pytest.mark.parametrize('legacy', [False, True])
def test_domain_migrations_preserve_data_and_enforce_permissions(database, legacy):
    dsn, passwords, environment = database
    if legacy:
        legacy_database(dsn)
    with psycopg.connect(dsn) as conn:
        old_oid = conn.execute("SELECT 'public.measurement'::regclass::oid").fetchone()[0] if legacy else None
        bootstrap.provision(conn, passwords)
    migrate(environment)
    # Repeat provisioning and migrations without moving or duplicating data.
    with psycopg.connect(dsn) as conn:
        bootstrap.provision(conn, passwords)
    migrate(environment)
    with psycopg.connect(dsn) as admin:
        assert admin.execute("SELECT to_regclass('public.measurement')").fetchone()[0] is None
        assert admin.execute("SELECT to_regclass('public.alembic_version')").fetchone()[0] is None
        if legacy:
            assert admin.execute("SELECT 'clio.measurement'::regclass::oid").fetchone()[0] == old_oid
            assert admin.execute('SELECT count(*) FROM clio.measurement').fetchone()[0] == 1
            assert admin.execute('SELECT count(*) FROM clio.solar_wind_aggregate_pending').fetchone()[0] == 1
            assert admin.execute('SELECT count(*) FROM clio.solar_wind_retired_hour').fetchone()[0] == 1
            assert admin.execute("SELECT statistics->>'retained' FROM clio.solar_wind_aggregate").fetchone()[0] == 'true'
        for domain in ('api', 'clio'):
            for migrator in (False, True):
                with runtime(dsn, domain, passwords, migrator) as conn:
                    other_table = 'clio.measurement' if domain == 'api' else 'api.dashboard_user'
                    with pytest.raises(psycopg.errors.InsufficientPrivilege):
                        conn.execute(f'SELECT * FROM {other_table}')
            with runtime(dsn, domain, passwords) as conn:
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    conn.execute(f'CREATE TABLE {domain}.forbidden(id int)')
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    conn.execute(f'SET ROLE argus_{domain}_migrator')
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    conn.execute(f'DELETE FROM {domain}.alembic_version')
    with runtime(dsn, 'api', passwords) as conn:
        identifier = conn.execute("INSERT INTO api.dashboard_user(username,password_hash,active) VALUES ('new','hash',true) RETURNING id").fetchone()[0]
        assert identifier == (2 if legacy else 1)
    with runtime(dsn, 'clio', passwords) as conn:
        # Trigger functions must resolve Clio tables even under another search path.
        conn.execute('SET search_path TO pg_catalog')
        conn.execute("""INSERT INTO clio.solar_wind_observation(kind, observed_at, spacecraft, active, received_at, "values", raw)
            VALUES ('mag', '2026-09-02 00:00:00+00', 'A', true, '2026-09-02 00:00:00+00', '{"bz": -5}', '{}')""")
        assert conn.execute("SELECT count(*) FROM clio.solar_wind_aggregate_pending WHERE hour='2026-09-02 00:00:00+00'").fetchone()[0] == 1
        if legacy:
            with pytest.raises(psycopg.errors.RaiseException):
                conn.execute("""INSERT INTO clio.solar_wind_observation(kind, observed_at, spacecraft, active, received_at, "values", raw)
                    VALUES ('mag', '2026-08-01 00:00:00+00', 'A', true, '2026-09-02 00:00:00+00', '{}', '{}')""")


def test_rejects_incomplete_legacy_state_without_partial_transfer(database):
    dsn, passwords, _ = database
    legacy_database(dsn)
    with psycopg.connect(dsn) as conn:
        conn.execute('DROP TABLE public.measurement')
    with pytest.raises(RuntimeError, match='Missing legacy table'):
        with psycopg.connect(dsn) as conn:
            bootstrap.provision(conn, passwords)
    with psycopg.connect(dsn) as conn:
        assert conn.execute("SELECT version_num FROM public.alembic_version").fetchone()[0] == bootstrap.LEGACY_HEAD
        assert conn.execute("SELECT to_regclass('public.solar_wind_observation')").fetchone()[0] is not None
        assert conn.execute("SELECT to_regclass('clio.solar_wind_observation')").fetchone()[0] is None


def test_scheduler_retries_restarts_and_serializes_jobs(database, monkeypatch):
    dsn, passwords, environment = database
    with psycopg.connect(dsn) as conn:
        bootstrap.provision(conn, passwords)
    migrate(environment)
    for key in ('DB_NAME', 'DB_HOST', 'DB_PORT', 'CLIO_DB_PASSWORD'):
        monkeypatch.setenv(key, environment[key])
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
    with runtime(dsn, 'clio', passwords) as conn:
        conn.execute('SELECT pg_advisory_lock(%s)', (JOBS['refresh'][1],))
        with pytest.raises(JobBusy):
            execute('refresh', lambda: calls.append(3), now=now)
    assert execute('refresh', lambda: calls.append(4), scheduled=True, now=now + timedelta(hours=1))
    assert calls == [1, 4]
