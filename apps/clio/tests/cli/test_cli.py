"""Clio parses one-shot commands and reports collection failures."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock

import pytest

from clio import cli
from clio.commands import runner
from common import runtime


def test_invoke_cleans_up_on_failure_without_changing_process_argv(monkeypatch):
    from common.db import session
    original = sys.argv
    args = SimpleNamespace(limit=12)
    command = AsyncMock(side_effect=ValueError('command failed'))
    dispose = AsyncMock()
    from clio.commands import normalize
    monkeypatch.setattr(normalize, 'run', command)
    monkeypatch.setattr(session, 'dispose_engine', dispose)
    with pytest.raises(ValueError, match='command failed'):
        runner.invoke('normalize', args)
    command.assert_awaited_once_with(args)
    dispose.assert_awaited_once()
    assert sys.argv is original


@pytest.mark.parametrize(('argv', 'command'), [
    (['collect-events'], 'collect-events'),
    (['backfill', '--from', '2026-08-31', '--to', '2026-09-01'], 'backfill'),
    (['normalize'], 'normalize'),
    (['collect', 'kp', 'dst'], 'collect'),
    (['collect', 'gong'], 'collect'),
    (['collect', 'sdo'], 'collect'),
    (['backfill', 'sdo'], 'backfill'),
    (['backfill', 'gong'], 'backfill'),
])
def test_manual_commands_invoke_operation_directly(monkeypatch, argv, command):
    invoke = Mock(return_value=None)
    monkeypatch.setattr(cli, 'invoke', invoke)
    monkeypatch.setattr(runtime, 'run_command', lambda run: run())
    cli.main(argv)
    assert invoke.call_args.args[0] == command


@pytest.mark.parametrize('argv', [['collect', '--help'], ['refresh', 'solar-wind', '--watch'],
                                  ['sdo-images', 'live'], ['schedule', 'live'], ['schedule', 'normalize'],
                                  ['status'], ['aggregate', '--limit', '0'], ['collect', 'files-backfill', '--unknown'],
                                  ['backfill', '--from', '2026-08-01', '--to', '2026-09-09'], ['worker'], ['check-health', 'worker'], ['crontab'], ['cron', 'install']])
def test_help_and_invalid_arguments_do_not_run(monkeypatch, argv):
    execute = Mock()
    monkeypatch.setattr(runtime, 'run_command', execute)
    with pytest.raises(SystemExit) as result:
        cli.main(argv)
    assert result.value.code == (0 if '--help' in argv else 2)
    execute.assert_not_called()


def test_configured_backfill_accepts_default_depth(monkeypatch):
    invoke = Mock(return_value=None)
    monkeypatch.setattr(cli, 'invoke', invoke)
    monkeypatch.setattr(runtime, 'run_command', lambda run: run())
    cli.main(['backfill', 'v', 'n', 't'])
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
    monkeypatch.setattr(runtime, 'run_command', lambda run: run())
    cli.main(['backfill'])
    args = invoke.call_args.args[1]
    assert args.metrics is None and args.start is None and args.end is None


@pytest.mark.parametrize('failed', [[], ['dst']])
def test_collection_exit_status_reflects_partial_failures(monkeypatch, failed):
    monkeypatch.setattr(cli, 'invoke', lambda *args: {'failed_metrics': failed, 'downloaded_measurements': 1})
    monkeypatch.setattr(runtime, 'run_command', lambda run: run())
    if failed:
        with pytest.raises(SystemExit) as result:
            cli.main(['collect', 'dst'])
        assert result.value.code == 1
    else:
        cli.main(['collect', 'dst'])
