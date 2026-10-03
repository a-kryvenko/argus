"""Clio parses once, validates before work and owns the manual-job locks."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock

import pytest

from clio import cli
from clio.scheduling import jobs as scheduler
from common import runtime


def test_invoke_cleans_up_on_failure_without_changing_process_argv(monkeypatch):
    from clio.db import session
    original = sys.argv
    args = SimpleNamespace(limit=12)
    command = AsyncMock(side_effect=ValueError('command failed'))
    dispose = AsyncMock()
    monkeypatch.setattr(cli.importlib, 'import_module', lambda name: SimpleNamespace(run=command))
    monkeypatch.setattr(session, 'dispose_engine', dispose)
    with pytest.raises(ValueError, match='command failed'):
        cli.invoke('normalize', args)
    command.assert_awaited_once_with(args)
    dispose.assert_awaited_once()
    assert sys.argv is original


@pytest.mark.parametrize(('argv', 'job', 'command'), [
    (['backfill', '--from', '2026-08-31', '--to', '2026-09-01'], 'refresh', 'backfill'),
    (['normalize'], 'refresh', 'normalize'),
    (['collect', 'kp', 'dst'], 'live', 'collect'),
    (['collect', 'aia193'], 'aia-live', 'collect'),
    (['backfill', 'aia193'], 'aia', 'backfill'),
])
def test_manual_commands_keep_their_job_lock(monkeypatch, argv, job, command):
    invoke = Mock(return_value=None)
    execute = Mock(side_effect=lambda name, run: run())
    monkeypatch.setattr(cli, 'invoke', invoke)
    monkeypatch.setattr(scheduler, 'execute', execute)
    monkeypatch.setattr(runtime, 'run_command', lambda run: run())
    cli.main(argv)
    assert execute.call_args.args[0] == job
    assert invoke.call_args.args[0] == command


@pytest.mark.parametrize('argv', [['collect', '--help'], ['refresh', 'solar-wind', '--watch'],
                                  ['schedule', 'live'], ['schedule', 'normalize'],
                                  ['aggregate', '--limit', '0'], ['collect', 'aia', '--unknown'],
                                  ['backfill', '--from', '2026-08-01', '--to', '2026-09-09']])
def test_help_and_invalid_arguments_do_not_run_or_lock(monkeypatch, argv):
    execute = Mock()
    monkeypatch.setattr(runtime, 'run_command', execute)
    with pytest.raises(SystemExit) as result:
        cli.main(argv)
    assert result.value.code == (0 if '--help' in argv else 2)
    execute.assert_not_called()


@pytest.mark.parametrize('command', ['check-health'])
def test_diagnostic_exit_status_is_preserved(monkeypatch, command):
    monkeypatch.setattr(cli, 'invoke', lambda name, args: 1)
    monkeypatch.setattr(runtime, 'run_command', lambda run: run())
    with pytest.raises(SystemExit) as error:
        cli.main([command, 'worker'] if command == 'check-health' else [command])
    assert error.value.code == 1


def test_configured_backfill_accepts_default_depth_and_keeps_lock(monkeypatch):
    invoke = Mock(return_value=None)
    execute = Mock(side_effect=lambda name, run: run())
    monkeypatch.setattr(cli, 'invoke', invoke)
    monkeypatch.setattr(scheduler, 'execute', execute)
    monkeypatch.setattr(runtime, 'run_command', lambda run: run())
    cli.main(['backfill', 'v', 'n', 't'])
    assert execute.call_args.args[0] == 'refresh'
    args = invoke.call_args.args[1]
    assert args.metrics == ['v', 'n', 't']
    assert args.start is None and args.end is None


@pytest.mark.parametrize('argv', [
    ['backfill', 'v', '--from', '2026-08-31'],
    ['backfill', 'aia171'],
])
def test_incomplete_backfill_arguments_fail_before_work(monkeypatch, argv):
    run = Mock()
    monkeypatch.setattr(runtime, 'run_command', run)
    with pytest.raises(SystemExit) as result:
        cli.main(argv)
    assert result.value.code == 2
    run.assert_not_called()


def test_backfill_without_arguments_uses_configured_observations(monkeypatch):
    invoke = Mock(return_value=None)
    monkeypatch.setattr(cli, 'invoke', invoke)
    monkeypatch.setattr(scheduler, 'execute', lambda name, run: run())
    monkeypatch.setattr(runtime, 'run_command', lambda run: run())
    cli.main(['backfill'])
    args = invoke.call_args.args[1]
    assert args.metrics is None and args.start is None and args.end is None
