import asyncio
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PROPHET = ROOT / 'apps/prophet'


def test_cli_help_is_available_without_loading_models():
    result = subprocess.run(
        [sys.executable, '-c', 'from prophet.cli import main; main()', '--help'],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'generate' in result.stdout and 'worker' not in result.stdout


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
assert importlib.util.find_spec('clio') is None
import prophet.db.session
import prophet.services.generation.calculation
from forecast_core.api import AtmosphericDensityForecastService
'''
    result = subprocess.run([str(python), '-c', code], cwd=tmp_path,
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_removed_aliases_are_rejected_before_starting_a_run(monkeypatch):
    import pytest
    from unittest.mock import Mock
    from prophet.services.generation import cycle
    from prophet import cli
    from prophet.services.releases import recording as ledger
    begin = Mock()
    monkeypatch.setattr(ledger.RunRecorder, 'begin', begin)
    for alias in ('wind', 'kp', 'density'):
        monkeypatch.setattr(sys, 'argv', ['prophet', 'generate', alias])
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2
        with pytest.raises(ValueError, match='Unsupported forecast product'):
            asyncio.run(cycle.generate(alias, writer=None))
    begin.assert_not_called()


def test_removed_export_command_is_rejected(monkeypatch):
    import pytest
    from prophet import cli
    monkeypatch.setattr(sys, 'argv', ['prophet', 'export'])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_refresh_alias_generates_selected_product(monkeypatch):
    from contextlib import nullcontext
    from unittest.mock import Mock
    from prophet import cli
    from prophet.db import locks as worker
    from prophet.services.generation import cycle
    from common import runtime
    from unittest.mock import AsyncMock
    generate = AsyncMock()
    monkeypatch.setattr(cycle, 'generate', generate)
    monkeypatch.setattr(worker, 'generation_lock', nullcontext)
    monkeypatch.setattr(runtime, 'run_command', lambda run: run())
    cli.main(['refresh', 'solar-wind-speed'])
    generate.assert_called_once_with('solar-wind-speed', writer=None)


def test_removed_inspection_commands_fail_before_loading_handlers(monkeypatch):
    import pytest
    from unittest.mock import Mock
    from prophet import cli
    handler = Mock()
    from common import runtime
    monkeypatch.setattr(runtime, 'run_command', handler)
    for command in ('worker', 'status', 'runs', 'show-run', 'slots', 'demo'):
        with pytest.raises(SystemExit) as error:
            cli.main([command])
        assert error.value.code == 2
    handler.assert_not_called()
