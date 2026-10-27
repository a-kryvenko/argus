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
    assert {'normalize', 'backfill', 'cleanup', 'sdo-live', 'collect-events', 'collect-enlil'} <= set(names)
    backfill = [call.args[1] for call in invoke.call_args_list if call.args[0] == 'backfill']
    assert len(backfill) == 1
    assert set(vars(backfill[0])) == {'command', 'now'}


def test_monitored_source_intervals_match_checked_in_cron():
    from clio.config import load_observation_config
    from clio.ingestion.numeric_live import SOURCE_IDS
    from clio.monitoring.specs import poll_seconds

    config = load_observation_config()
    cron = Path(__file__).resolve().parents[2] / 'cronjobs.txt'
    checked = set()
    for line in cron.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        fields = line.split(maxsplit=5)
        args = shlex.split(fields[5])
        if args[3] != 'collect':
            continue
        for metric in args[4:]:
            policy = config.observations.get(metric)
            for product in policy.sources.live if policy else []:
                source = SOURCE_IDS.get(product)
                if source is None:
                    continue
                seconds = poll_seconds(source)
                minute = '*' if seconds == 60 else f'*/{int(seconds // 60)}'
                assert fields[:5] == [minute, '*', '*', '*', '*']
                checked.add(source)
    assert checked == set(SOURCE_IDS.values())
