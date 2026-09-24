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


def test_installed_environment_imports_without_observation_storage(tmp_path):
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
assert importlib.util.find_spec('argus_clio') is None
import argus_prophet.db.session
import argus_prophet.services.generation.calculation
import argus_prophet.services.density.forecast
'''
    result = subprocess.run([str(python), '-c', code], cwd=tmp_path,
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_removed_aliases_are_rejected_before_starting_a_run(monkeypatch):
    import pytest
    from unittest.mock import Mock
    from argus_prophet.services.generation import cycle
    from argus_prophet import cli
    from argus_prophet.services import runs as ledger
    begin = Mock()
    monkeypatch.setattr(ledger.RunRecorder, 'begin', begin)
    for alias in ('wind', 'kp', 'density'):
        monkeypatch.setattr(sys, 'argv', ['prophet', 'generate', alias])
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2
        with pytest.raises(ValueError, match='Unsupported forecast product'):
            cycle.generate(alias)
    begin.assert_not_called()


def test_removed_export_command_is_rejected(monkeypatch):
    import pytest
    from argus_prophet import cli
    monkeypatch.setattr(sys, 'argv', ['prophet', 'export'])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
