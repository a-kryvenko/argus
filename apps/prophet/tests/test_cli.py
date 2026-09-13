import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PROPHET = ROOT / 'apps/prophet'


def test_cli_help_is_available_without_loading_models():
    result = subprocess.run(
        [sys.executable, '-c', 'from argus_prophet.cli import main; main()', '--help'],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'generate' in result.stdout and 'worker' in result.stdout


def test_installed_environment_imports_without_api_or_database(tmp_path):
    import pytest
    python = PROPHET / '.venv/bin/python'
    if not python.exists():
        pytest.skip('Sync the standalone Prophet environment to check installed imports')
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment['ARGUS_WORKDIR'] = str(ROOT)
    code = '''
import importlib.util
assert importlib.util.find_spec('app') is None
assert importlib.util.find_spec('sqlalchemy') is None
assert importlib.util.find_spec('psycopg') is None
import argus_prophet.commands.generate_forecast
import argus_prophet.commands.generate_atmospheric_density_forecast
'''
    result = subprocess.run([str(python), '-c', code], cwd=tmp_path,
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
