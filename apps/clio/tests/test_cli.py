"""Clio parses once, validates before work and owns the manual-job locks."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock

import pytest

from argus_clio import cli, scheduler
from common import runtime


def test_invoke_cleans_up_on_failure_without_changing_process_argv(monkeypatch):
    from argus_clio.db import session
    original = sys.argv
    args = SimpleNamespace(limit=12)
    command = AsyncMock(side_effect=ValueError('command failed'))
    dispose = AsyncMock()
    monkeypatch.setattr(cli.importlib, 'import_module', lambda name: SimpleNamespace(run=command))
    monkeypatch.setattr(session, 'dispose_engine', dispose)
    with pytest.raises(ValueError, match='command failed'):
        cli.invoke('aggregate', args)
    command.assert_awaited_once_with(args)
    dispose.assert_awaited_once()
    assert sys.argv is original


@pytest.mark.parametrize(('argv', 'job', 'command'), [
    (['backfill', '--from', '2026-08-31', '--to', '2026-09-01'], 'refresh', 'backfill'),
    (['refresh', 'observations'], 'refresh', 'observations'),
    (['aggregate', '--limit', '12'], 'aggregate', 'aggregate'),
    (['collect', 'aia', '--history-days', '1'], 'aia', 'aia'),
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


def test_refresh_collects_all_once_and_stops_on_failure(monkeypatch):
    invoke = Mock(return_value=None)
    monkeypatch.setattr(cli, 'invoke', invoke)
    monkeypatch.setattr(scheduler, 'execute', lambda name, run: run())
    monkeypatch.setattr(runtime, 'run_command', lambda run: run())
    cli.main(['refresh'])
    assert [call.args[0] for call in invoke.call_args_list] == ['solar-wind', 'geomagnetic', 'observations']
    invoke.reset_mock()
    invoke.side_effect = ValueError('source failed')
    with pytest.raises(ValueError, match='source failed'):
        cli.main(['refresh'])
    assert invoke.call_count == 1


@pytest.mark.parametrize('argv', [['refresh', '--help'], ['refresh', 'solar-wind', '--watch'],
                                  ['aggregate', '--limit', '0'], ['collect', 'aia', '--unknown'],
                                  ['backfill', '--from', '2026-08-01', '--to', '2026-09-09']])
def test_help_and_invalid_arguments_do_not_run_or_lock(monkeypatch, argv):
    execute = Mock()
    monkeypatch.setattr(runtime, 'run_command', execute)
    with pytest.raises(SystemExit) as result:
        cli.main(argv)
    assert result.value.code == (0 if '--help' in argv else 2)
    execute.assert_not_called()


def test_schedule_refresh_only_normalizes(monkeypatch):
    invoke = Mock(return_value=None)
    monkeypatch.setattr(cli, 'invoke', invoke)
    monkeypatch.setattr(scheduler, 'work', lambda name, run: run())
    monkeypatch.setattr(runtime, 'run_command', lambda run: run())
    cli.main(['schedule', 'refresh'])
    assert invoke.call_args.args[0] == 'observations'


@pytest.mark.parametrize('command', ['audit', 'cleanup', 'check-health'])
def test_diagnostic_exit_status_is_preserved(monkeypatch, command):
    monkeypatch.setattr(cli, 'invoke', lambda name, args: 1)
    monkeypatch.setattr(runtime, 'run_command', lambda run: run())
    with pytest.raises(SystemExit) as error:
        cli.main([command, 'worker'] if command == 'check-health' else [command])
    assert error.value.code == 1
