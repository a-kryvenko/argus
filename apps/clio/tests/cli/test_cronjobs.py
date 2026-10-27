"""Every checked-in cron job must invoke a supported one-shot command."""
from pathlib import Path
import shlex
from unittest.mock import Mock

from clio import cli
from common import runtime


def test_cronjobs_invoke_supported_commands_without_docker(monkeypatch):
    invoke = Mock(return_value={})
    monkeypatch.setattr(cli, 'invoke', invoke)
    monkeypatch.setattr(runtime, 'run_command', lambda run: run())
    names = []
    source = Path(__file__).resolve().parents[2] / 'cronjobs.txt'
    for line in source.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        args = shlex.split(line.split(maxsplit=5)[5])
        assert args[:2] == ['@ROOT@/scripts/cron-run', 'clio']
        names.append(args[2])
        cli.main(args[3:])
    assert len(names) == len(set(names)) == invoke.call_count
    assert {'normalize', 'sdo-live', 'sdo-warmup', 'sdo-cleanup', 'collect-events', 'collect-enlil'} <= set(names)
    warmup = next(call.args[1] for call in invoke.call_args_list
                  if call.args[0] == 'backfill' and call.args[1].metrics == ['sdo'])
    assert warmup.batch is True
