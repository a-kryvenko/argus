"""Separate databases on an explicitly disposable PostgreSQL server.

Never reads project .env credentials. Creates/drops only UUID-named databases
and owners. Run serially; TEST_DATABASE_ADMIN_DSN must be administrative.
"""
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

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


provisioning = load(ROOT / 'scripts/db/provision.py')


def database_environment(urls):
    settings = {}
    for domain, value in urls.items():
        url = make_url(value)
        for field, value in dict(HOST=url.host, PORT=str(url.port or 5432), NAME=url.database,
                                 USER=url.username, PASSWORD=url.password).items():
            settings[domain.upper() + '_DB_' + field] = value
    return settings


@contextmanager
def provisioned_database(tmp_path):
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
                   'DEBUG': 'true', 'SENTRY_COLLECT_POINT': '', 'SENTRY_DSN': ''}
    try:
        provisioning.provision(admin_dsn, urls, apply=True)
        yield dsns, urls, environment
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as admin:
            for url in targets.values():
                admin.execute(sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)').format(sql.Identifier(url.database)))
                admin.execute(sql.SQL('DROP ROLE IF EXISTS {}').format(sql.Identifier(url.username)))


@pytest.fixture
def database(tmp_path):
    with provisioned_database(tmp_path) as settings:
        yield settings


def migrate(environment):
    for command in ([sys.executable, '-c', 'from clio.cli import main; main()', 'migrate', 'upgrade', 'head'],
                    [sys.executable, '-m', 'alembic', '-c', str(ROOT / 'apps/api/alembic.ini'), 'upgrade', 'head'],
                    [sys.executable, '-c', 'from argus_prophet.cli import main; main()', 'migrate', 'upgrade', 'head'],
                    [sys.executable, '-c', 'from argus_intelligence.cli import main; main()', 'migrate', 'upgrade', 'head']):
        result = subprocess.run(command, env=environment, cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def runtime(dsns, domain, urls):
    return psycopg.connect(urls[domain], autocommit=True)


