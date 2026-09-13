import os
from pathlib import Path
import subprocess
import pytest

from app.db.session import get_database_url


def test_api_cannot_fall_back_to_administrative_credentials(monkeypatch):
    monkeypatch.setenv('DB_NAME', 'test')
    monkeypatch.setenv('DB_USER', 'postgres')
    monkeypatch.setenv('DB_PASSWORD', 'admin-password')
    monkeypatch.delenv('API_DB_PASSWORD', raising=False)
    with pytest.raises(RuntimeError, match='API_DB_PASSWORD'):
        get_database_url()
    monkeypatch.setenv('API_DB_PASSWORD', 'runtime')
    monkeypatch.setenv('API_MIGRATION_PASSWORD', 'migration')
    assert get_database_url().username == 'argus_api'
    assert get_database_url().password == 'runtime'
    assert get_database_url(migration=True).username == 'argus_api_migrator'
    assert get_database_url(migration=True).password == 'migration'


def test_installed_api_has_no_collector_or_private_backend():
    root = Path(__file__).resolve().parents[1]
    python = root / '.venv/bin/python'
    if not python.exists():
        pytest.skip('Sync API environment to verify installed imports')
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    result = subprocess.run([str(python), '-c', '''
import importlib.util
assert importlib.util.find_spec('clio') is None
assert importlib.util.find_spec('argus_clio') is None
assert importlib.util.find_spec('forecast_core') is None
import app.main
'''], cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
