"""Provisioning and migration invariants shared by all services."""
import importlib.util
from pathlib import Path

import pytest

from tests.support.migrations import migration_sql

ROOT = Path(__file__).resolve().parents[2]


def test_provisioning_rejects_shared_databases_before_connecting(monkeypatch):
    pytest.importorskip('psycopg')
    spec = importlib.util.spec_from_file_location('provisioning', ROOT / 'scripts/db/provision.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.psycopg, 'connect', lambda *a, **kw: pytest.fail('Invalid plan connected to PostgreSQL'))
    admin = 'postgresql://admin:password@localhost/postgres'
    with pytest.raises(ValueError, match='separate database'):
        module.provision(admin, {'api': 'postgresql://api:pass@localhost/shared',
                                 'clio': 'postgresql://clio:pass@localhost/shared'}, apply=True)
    with pytest.raises(ValueError, match='separate owner'):
        module.provision(admin, {'api': 'postgresql://owner:pass@localhost/api',
                                 'clio': 'postgresql://owner:pass@localhost/clio'}, apply=True)
    with pytest.raises(ValueError, match='administrative connection server'):
        module.provision(admin, {'api': 'postgresql://api:pass@other-server/api'}, apply=True)


@pytest.mark.parametrize('domain', ['api', 'clio', 'prophet', 'intelligence'])
def test_schema_migrations_compile_for_a_single_owner(domain, tmp_path):
    result = migration_sql(domain, '20261027_file_observation' if domain == 'clio' else 'head', tmp_path)
    assert result.returncode == 0, result.stderr
    assert f'CREATE SCHEMA IF NOT EXISTS {domain}' in result.stdout
    assert f'{domain}.alembic_version' in result.stdout
    assert 'argus_' + domain + '_migrator' not in result.stdout
    assert 'REVOKE' not in result.stdout
