"""Clio parses one-shot commands and reports collection failures."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock

import pytest

from clio import cli
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
        cli.invoke('normalize', args)
    command.assert_awaited_once_with(args)
    dispose.assert_awaited_once()
    assert sys.argv is original


@pytest.mark.parametrize(('argv', 'command'), [
    (['collect', 'donki'], 'collect'),
    (['collect', 'enlil'], 'collect'),
    (['collect', 'donki', 'enlil', 'gong'], 'collect'),
    (['backfill'], 'backfill'),
    (['cleanup'], 'cleanup'),
    (['normalize'], 'normalize'),
    (['collect', 'kp', 'dst'], 'collect'),
    (['collect', 'gong'], 'collect'),
    (['collect', 'sdo'], 'collect'),
])
def test_manual_commands_invoke_operation_directly(monkeypatch, argv, command):
    invoke = Mock(return_value=None)
    monkeypatch.setattr(cli, 'invoke', invoke)
    monkeypatch.setattr(runtime, 'run_command', lambda run: run())
    cli.main(argv)
    assert invoke.call_args.args[0] == command


@pytest.mark.parametrize('argv', [['collect', '--help'], ['refresh', 'solar-wind', '--watch'],
                                  ['sdo-images', 'live'], ['schedule', 'live'], ['schedule', 'normalize'],
                                  ['collect-events'], ['collect-enlil'], ['backfill', 'donki'], ['backfill', 'enlil'],
                                  ['status'], ['aggregate', '--limit', '0'], ['collect', 'files-backfill', '--unknown'],
                                  ['backfill', '--from', '2026-08-01', '--to', '2026-09-09'], ['worker'], ['check-health', 'worker'], ['crontab'], ['cron', 'install']])
def test_help_and_invalid_arguments_do_not_run(monkeypatch, argv):
    execute = Mock()
    monkeypatch.setattr(runtime, 'run_command', execute)
    with pytest.raises(SystemExit) as result:
        cli.main(argv)
    assert result.value.code == (0 if '--help' in argv else 2)
    execute.assert_not_called()


@pytest.mark.parametrize('argv', [
    ['backfill', 'v', '--from', '2026-08-31'],
    ['backfill', 'aia171'],
    ['backfill', 'sdo'],
    ['backfill', 'gong'],
    ['backfill', '--batch'],
    ['backfill', '--from', '2026-08-31', '--to', '2026-09-01'],
    ['sdo-cleanup'],
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
    assert set(vars(args)) == {'command', 'now'}


def test_serve_runs_observation_api(monkeypatch):
    import uvicorn
    serve = Mock()
    monkeypatch.setattr(uvicorn, 'run', serve)
    monkeypatch.setattr(runtime, 'run_command', lambda run: run())
    cli.main(['serve', '--host', '127.0.0.1', '--port', '9000'])
    serve.assert_called_once_with('clio.main:app', host='127.0.0.1', port=9000)


@pytest.mark.parametrize('name', ['collect', 'backfill'])
@pytest.mark.parametrize('report,code', [
    ({'status': 'complete', 'failed_metrics': []}, 0),
    ({'status': 'partial', 'failed_metrics': [], 'missing_observed_slots': {'v': 12}}, 1),
    ({'status': 'failed'}, 1),
    ({'failed_metrics': ['v']}, 1),
    ({'failed': 1}, 1),
])
def test_exit_status_includes_incomplete_results(monkeypatch, name, report, code):
    monkeypatch.setattr(cli, 'invoke', Mock(return_value=report))
    monkeypatch.setattr(runtime, 'run_command', lambda run: run())
    if code:
        with pytest.raises(SystemExit) as result:
            cli.main([name])
        assert result.value.code == code
    else:
        cli.main([name])


def test_invoke_preserves_successful_report_and_releases_connections(monkeypatch):
    from clio.commands import backfill
    from clio import config
    from common.db import session
    report = {'status': 'complete', 'failed_metrics': []}
    args = SimpleNamespace(now=object())
    run = AsyncMock(return_value=report)
    dispose = AsyncMock()
    monkeypatch.setattr(backfill, 'run', run)
    monkeypatch.setattr(session, 'dispose_engine', dispose)
    load = Mock(side_effect=AssertionError('CLI must let the handler load its config'))
    monkeypatch.setattr(config, 'load_observation_config', load)
    assert cli.invoke('backfill', args) is report
    run.assert_awaited_once_with(args)
    dispose.assert_awaited_once()
    load.assert_not_called()
