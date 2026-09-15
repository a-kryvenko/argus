"""Service connections use isolated settings and preserve raw passwords."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    'api': 'apps/api/app/db/session.py',
    'clio': 'apps/clio/src/argus_clio/db/session.py',
    'prophet': 'apps/prophet/src/argus_prophet/db/session.py',
    'intelligence': 'apps/intelligence/src/argus_intelligence/db.py',
}


@pytest.mark.parametrize('domain', PATHS)
def test_raw_passwords_and_domain_settings_are_isolated(domain, monkeypatch):
    from sqlalchemy.engine import make_url
    spec = importlib.util.spec_from_file_location('storage_' + domain, ROOT / PATHS[domain])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    get_url = getattr(module, 'get_database_url', None) or module.database_url
    prefix = domain.upper() + '_DB_'
    for field in ('NAME', 'USER', 'PASSWORD', 'HOST', 'PORT'):
        monkeypatch.delenv(prefix + field, raising=False)
    monkeypatch.setenv('DB_USER', 'postgres')
    monkeypatch.setenv('DB_PASSWORD', 'admin-secret')
    monkeypatch.setenv('DATABASE_URL', 'postgresql://admin:secret@db/admin')
    with pytest.raises(RuntimeError, match=prefix + 'PASSWORD'):
        get_url()
    for field, value in dict(NAME=domain, USER='owner', PASSWORD='temporary', PORT='5433').items():
        monkeypatch.setenv(prefix + field, value)
    for password in ('p@ss/word', 'a:b#c?d%20+e', 'space and $dollar', 'пароль=тест'):
        monkeypatch.setenv(prefix + 'PASSWORD', password)
        url = get_url()
        assert url.database == domain and url.username == 'owner'
        assert url.host == 'localhost' and url.port == 5433
        assert url.password == password
        assert make_url(url.render_as_string(hide_password=False)).password == password
    for invalid in ('notaport', '0', '65536'):
        monkeypatch.setenv(prefix + 'PORT', invalid)
        with pytest.raises(RuntimeError, match=prefix + 'PORT') as exc:
            get_url()
        assert password not in str(exc.value)


def test_provisioning_rejects_shared_databases_before_connecting(monkeypatch):
    pytest.importorskip('psycopg')
    spec = importlib.util.spec_from_file_location('provisioning', ROOT / 'scripts/provision-databases.py')
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


@pytest.mark.parametrize('domain', PATHS)
def test_full_migration_chain_compiles_for_a_single_owner(domain, tmp_path):
    import os
    import subprocess
    import sys
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/project.yaml').write_text('project: {name: test}\n')
    (tmp_path / 'configs/models_registry.yaml').write_text('models: {}\n')
    environment = {**os.environ, 'ARGUS_WORKDIR': str(tmp_path),
                   **{domain.upper() + '_DB_' + key: value for key, value in
                      dict(HOST='localhost', PORT='5432', NAME=domain, USER='owner', PASSWORD='p@ss:word/%').items()},
                   'PYTHONPATH': ':'.join(str(ROOT / p) for p in (
                       'apps/api', 'apps/clio/src', 'apps/prophet/src', 'apps/intelligence/src',
                       'packages/common/src', 'packages/clio/src', 'packages/forecast/src'))}
    if domain == 'api':
        command = [sys.executable, '-m', 'alembic', '-c', str(ROOT / 'apps/api/alembic.ini')]
    else:
        command = [sys.executable, '-c', f'from argus_{domain}.cli import main; main()', 'migrate']
    result = subprocess.run([*command, 'upgrade', 'head', '--sql'], env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f'CREATE SCHEMA IF NOT EXISTS {domain}' in result.stdout
    assert f'{domain}.alembic_version' in result.stdout
    assert 'argus_' + domain + '_migrator' not in result.stdout
    assert 'REVOKE' not in result.stdout
