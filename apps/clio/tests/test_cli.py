"""Dispatch must preserve process arguments and the shared manual-job locks."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from argus_clio import cli, scheduler
from argus_clio.commands import _runner


def test_invoke_passes_arguments_without_changing_process_argv(monkeypatch):
    original = sys.argv

    def command(argv):
        assert argv == ['--limit', '12']
        assert sys.argv is original
        raise ValueError('command failed')

    monkeypatch.setattr(cli.importlib, 'import_module', lambda name: SimpleNamespace(main=command))
    with pytest.raises(ValueError, match='command failed'):
        cli.invoke('aggregate', ['--limit', '12'])
    assert sys.argv is original


@pytest.mark.parametrize(('argv', 'job', 'command', 'arguments'), [
    (['backfill', '--from', '2026-08-31', '--to', '2026-09-01'], 'refresh', 'backfill',
     ['--from', '2026-08-31', '--to', '2026-09-01']),
    (['refresh'], 'refresh', 'refresh', []),
    (['aggregate', '--limit', '12'], 'aggregate', 'aggregate', ['--limit', '12']),
    (['collect', 'aia', '--history-days', '1'], 'aia', 'aia', ['--history-days', '1']),
])
def test_manual_commands_keep_their_job_lock(monkeypatch, argv, job, command, arguments):
    invoke = Mock()
    execute = Mock(side_effect=lambda name, run: run())
    monkeypatch.setattr(cli, 'invoke', invoke)
    monkeypatch.setattr(scheduler, 'execute', execute)
    monkeypatch.setattr(_runner, 'run_command', lambda run: run())
    cli.main(argv)
    assert execute.call_args.args[0] == job
    invoke.assert_called_once_with(command, arguments)


def test_refresh_help_does_not_run_or_lock(monkeypatch, capsys):
    from argus_clio.commands import refresh_observations
    refresh = Mock()
    execute = Mock()
    monkeypatch.setattr(refresh_observations, '_refresh', refresh)
    monkeypatch.setattr(scheduler, 'execute', execute)
    monkeypatch.setattr(_runner, 'run_command', lambda run: run())
    with pytest.raises(SystemExit) as result:
        cli.main(['refresh', '--help'])
    assert result.value.code == 0
    assert 'usage:' in capsys.readouterr().out
    refresh.assert_not_called()
    execute.assert_not_called()
