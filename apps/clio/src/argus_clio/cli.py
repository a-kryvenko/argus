"""Clio command parsing and execution; handlers only perform their operation."""
import argparse
import asyncio
import importlib
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from argus_clio.commands._arguments import utc_hour, boundary

COMMANDS = {
    'solar-wind': 'collect_solar_wind', 'geomagnetic': 'collect_geomagnetic', 'aia': 'collect_aia',
    'backfill': 'backfill_observations', 'observations': 'refresh_observations', 'aggregate': 'aggregate_solar_wind',
    'audit': 'audit_solar_wind', 'cleanup': 'cleanup_solar_wind', 'check-health': 'check_collector_health',
}


def invoke(name, args):
    module = importlib.import_module(f'argus_clio.commands.{COMMANDS[name]}')
    if name == 'check-health':
        return module.run(args)
    from argus_clio.db.session import dispose_engine
    async def execute():
        try:
            return await module.run(args)
        finally:
            await dispose_engine()
    return asyncio.run(execute())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(watch=False, history_days=40, limit=240)
    commands = parser.add_subparsers(dest='command', required=True)
    serve = commands.add_parser('serve')
    serve.add_argument('--host', default='0.0.0.0')
    serve.add_argument('--port', type=int, default=8000)
    collect = commands.add_parser('collect')
    sources = collect.add_subparsers(dest='source', required=True)
    for name in ('solar-wind', 'geomagnetic'):
        sources.add_parser(name).add_argument('--watch', action='store_true')
    sources.add_parser('aia').add_argument('--history-days', type=int, default=40)
    refresh = commands.add_parser('refresh', help='Collect sources and refresh normalized observations once')
    refresh.add_argument('source', nargs='?', default='all', choices=['all', 'observations', 'solar-wind', 'geomagnetic'])
    commands.add_parser('aggregate').add_argument('--limit', type=int, default=240)
    backfill = commands.add_parser('backfill')
    backfill.add_argument('--from', dest='start', type=boundary, required=True)
    backfill.add_argument('--to', dest='end', type=boundary, required=True)
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
    commands.add_parser('schedule').add_argument('job', choices=['refresh', 'aggregate', 'aia'])
    args, remainder = parser.parse_known_args(argv)
    if args.command != 'migrate' and remainder:
        parser.error('Unrecognized arguments: ' + ' '.join(remainder))
    args.now = datetime.now(UTC)
    if args.command == 'audit':
        args.end = args.end or args.now.replace(minute=0, second=0, microsecond=0)
        args.start = args.start or args.end - timedelta(days=7)
        if args.retention_days < 1 or args.detail_limit < 1:
            parser.error('--retention-days and --detail-limit must be positive')
    if args.command in ('audit', 'backfill'):
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
    if args.command == 'check-health':
        return invoke('check-health', args)
    if args.command == 'status':
        from argus_clio.db.session import get_session_factory, dispose_engine
        from argus_clio.services.collection.status import source_status
        async def read_status():
            try:
                async with get_session_factory()() as session:
                    return await source_status(session)
            finally:
                await dispose_engine()
        print(json.dumps(asyncio.run(read_status()), default=str, indent=2))
    elif args.command == 'serve':
        import uvicorn
        uvicorn.run('argus_clio.main:app', host=args.host, port=args.port)
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
    elif args.command == 'worker':
        from argus_clio.worker import work
        work()
    elif args.command == 'schedule':
        from argus_clio.scheduler import work
        name = 'observations' if args.job == 'refresh' else args.job
        work(args.job, lambda: invoke(name, args))
    elif args.command == 'refresh':
        from argus_clio.scheduler import execute as locked
        for name in ('solar-wind', 'geomagnetic', 'observations') if args.source == 'all' else (args.source,):
            if name == 'observations':
                locked('refresh', lambda: invoke(name, args))
            else:
                invoke(name, args)
    else:
        from argus_clio.scheduler import execute as locked
        name = args.source if args.command == 'collect' else args.command
        job = {'backfill': 'refresh', 'aggregate': 'aggregate', 'aia': 'aia'}.get(name)
        if job:
            locked(job, lambda: invoke(name, args))
            return
        return invoke(name, args)


if __name__ == '__main__':
    main()
