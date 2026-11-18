"""Provisioning and migration invariants shared by all services."""
import pytest

from tests.support.migrations import migration_sql

@pytest.mark.parametrize('domain', ['api', 'clio', 'prophet', 'intelligence'])
def test_schema_migrations_compile_for_a_single_owner(domain, tmp_path):
    result = migration_sql(domain, '20261027_file_observation' if domain == 'clio' else 'head', tmp_path)
    assert result.returncode == 0, result.stderr
    assert f'CREATE SCHEMA IF NOT EXISTS {domain}' in result.stdout
    assert f'{domain}.alembic_version' in result.stdout
    assert 'argus_' + domain + '_migrator' not in result.stdout
    assert 'REVOKE' not in result.stdout
