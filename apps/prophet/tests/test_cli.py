import subprocess
import sys

import pytest


def test_cli_help_is_available_without_loading_models():
    result = subprocess.run(
        [sys.executable, '-c', 'from prophet.cli import main; main()', '--help'],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert '{serve,generation,verification,cleanup}' in result.stdout


def test_generation_generates_selected_product(monkeypatch):
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
    cli.main(['generation', 'solar-wind-speed'])
    generate.assert_called_once_with('solar-wind-speed', writer=None)


@pytest.mark.parametrize('failure', [None, RuntimeError('failed'), SystemExit(130)])
def test_invoke_disposes_database_after_command(monkeypatch, failure):
    from argparse import Namespace
    from unittest.mock import AsyncMock
    from common.db import session
    from prophet import cli
    from prophet.commands import cleanup

    run = AsyncMock(return_value=None, side_effect=failure)
    dispose = AsyncMock()
    monkeypatch.setattr(cleanup, 'run', run)
    monkeypatch.setattr(session, 'dispose_engine', dispose)
    args = Namespace(days=90, apply=False)
    if failure is None:
        cli.invoke('cleanup', args)
    else:
        with pytest.raises(type(failure)):
            cli.invoke('cleanup', args)
    run.assert_awaited_once_with(args)
    dispose.assert_awaited_once()


@pytest.mark.parametrize('arguments', [
    ['unknown-command'], ['verification', '--unknown-option'],
    ['generation', 'unknown-product'],
])
def test_invalid_arguments_fail_before_loading_handlers(monkeypatch, arguments):
    from unittest.mock import Mock
    from common import runtime
    from prophet import cli

    run = Mock()
    monkeypatch.setattr(runtime, 'run_command', run)
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2
    run.assert_not_called()
