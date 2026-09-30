"""Clio command parsing and execution; handlers only perform their operation."""
import argparse
import asyncio
import importlib
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from clio.commands._arguments import utc_hour, boundary
from clio.ingestion.products import OBSERVATIONS

COMMANDS = {
    'sdo-images': 'sdo_images',
    'native-wind': 'collect_solar_wind',
    'backfill': 'backfill_observations', 'normalize': 'normalize', 'aggregate': 'aggregate_solar_wind',
    'collect': 'collect',
    'audit': 'audit_solar_wind', 'cleanup': 'cleanup_solar_wind', 'check-health': 'check_collector_health',
}


def invoke(name, args):
    module = importlib.import_module(f'clio.commands.{COMMANDS[name]}')
    if name == 'check-health':
        return module.run(args)
    from clio.db.session import dispose_engine
    async def execute():
        try:
            return await module.run(args)
        finally:
            await dispose_engine()
    return asyncio.run(execute())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    serve = commands.add_parser('serve')
    serve.add_argument('--host', default='0.0.0.0')
    serve.add_argument('--port', type=int, default=8000)
    commands.add_parser('collect').add_argument('metrics', nargs='*')
    commands.add_parser('normalize')
    commands.add_parser('aggregate').add_argument('--limit', type=int, default=240)
    backfill = commands.add_parser('backfill')
    backfill.add_argument('--from', dest='start', type=boundary)
    backfill.add_argument('--to', dest='end', type=boundary)
    backfill.add_argument('metrics', nargs='*',
                          help='Use configured gap-only backfill (defaults to each metric history depth)')
    audit = commands.add_parser('audit')
    audit.add_argument('--from', dest='start', type=utc_hour)
    audit.add_argument('--to', dest='end', type=utc_hour)
    audit.add_argument('--retention-days', type=int, default=90)
    audit.add_argument('--detail-limit', type=int, default=200)
    audit.add_argument('--json', action='store_true')
    cleanup = commands.add_parser('cleanup')
    cleanup.add_argument('--from', dest='start', type=utc_hour)
    cleanup.add_argument('--apply', action='store_true')
    cleanup.add_argument('--retention-days', type=int, default=90)
    cleanup.add_argument('--limit', type=int, default=24)
    cleanup.add_argument('--json', action='store_true')
    commands.add_parser('check-health').add_argument('collector', choices=['solar-wind', 'geomagnetic', 'worker'])
    commands.add_parser('migrate', add_help=False)
    commands.add_parser('status')
    commands.add_parser('worker')
    commands.add_parser('sdo-images').add_argument('mode', choices=['live', 'warmup', 'cleanup'])
    args, remainder = parser.parse_known_args(argv)
    if args.command != 'migrate' and remainder:
        parser.error('Unrecognized arguments: ' + ' '.join(remainder))
    args.now = datetime.now(UTC)
    if args.command in ('collect', 'backfill'):
        if any(m not in OBSERVATIONS for m in args.metrics):
            parser.error('Unknown observation metric')
        args.metrics = args.metrics or None
    if args.command == 'audit':
        args.end = args.end or args.now.replace(minute=0, second=0, microsecond=0)
        args.start = args.start or args.end - timedelta(days=7)
        if args.retention_days < 1 or args.detail_limit < 1:
            parser.error('--retention-days and --detail-limit must be positive')
    if args.command == 'backfill':
        if (args.start is None) != (args.end is None):
            parser.error('--from and --to must be specified together')
    if args.command == 'audit' or (args.command == 'backfill' and args.start is not None):
        if not timedelta(0) < args.end - args.start <= timedelta(days=31) or args.end > args.now:
            parser.error('Choose a past range of at most 31 days, with --from before --to')
    if args.command == 'cleanup' and (args.retention_days < 90 or not 1 <= args.limit <= 240):
        parser.error('--retention-days must be at least 90; --limit must be 1–240')
    if args.command == 'aggregate' and args.limit < 1:
        parser.error('--limit must be positive')
    from common.runtime import run_command
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    result = run_command(lambda: execute(args, remainder))
    if result:
        raise SystemExit(result)


def execute(args, remainder):
    if args.command not in ('migrate', 'check-health'):
        from clio.config import load_observation_config
        load_observation_config()
    if args.command == 'check-health':
        return invoke('check-health', args)
    if args.command == 'status':
        from clio.db.session import get_session_factory, dispose_engine
        from clio.monitoring.status import source_status
        async def read_status():
            try:
                async with get_session_factory()() as session:
                    return await source_status(session)
            finally:
                await dispose_engine()
        print(json.dumps(asyncio.run(read_status()), default=str, indent=2))
    elif args.command == 'serve':
        import uvicorn
        uvicorn.run('clio.main:app', host=args.host, port=args.port)
    elif args.command == 'migrate':
        from alembic.config import CommandLine, Config
        cli = CommandLine(prog='clio migrate')
        if not remainder:
            cli.parser.print_help()
            return
        options = cli.parser.parse_args(remainder)
        config = Config()
        config.cmd_opts = options
        config.set_main_option('script_location', str(Path(__file__).parent / 'migrations'))
        cli.run_cmd(config, options)
    elif args.command == 'sdo-images':
        print(json.dumps(invoke('sdo-images', args)))
    elif args.command == 'worker':
        from clio.worker import work
        work()
    else:
        from clio.scheduling.jobs import execute as locked
        name = args.command
        job = {'backfill': 'refresh', 'collect': 'live', 'aggregate': 'aggregate', 'normalize': 'refresh'}.get(name)
        if name in ('backfill', 'collect') and args.metrics and all(OBSERVATIONS[m].kind == 'file' for m in args.metrics):
            job = 'aia' if name == 'backfill' else 'aia-live'
        if job:
            locked(job, lambda: invoke(name, args))
            return
        return invoke(name, args)


if __name__ == '__main__':
    main()
