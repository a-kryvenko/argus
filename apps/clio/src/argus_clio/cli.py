"""Installed Clio service, collector, scheduler and migration entry points."""
import argparse
import importlib
import logging
from pathlib import Path
import sys


COMMANDS = {
    'solar-wind': 'collect_solar_wind', 'geomagnetic': 'collect_geomagnetic',
    'refresh': 'refresh_observations', 'aggregate': 'aggregate_solar_wind',
    'audit': 'audit_solar_wind', 'cleanup': 'cleanup_solar_wind',
    'check-health': 'check_collector_health',
}


def invoke(name, arguments=()):
    previous = sys.argv
    sys.argv = [f'clio {name}', *arguments]
    try:
        importlib.import_module(f'argus_clio.commands.{COMMANDS[name]}').main()
    finally:
        sys.argv = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    serve = commands.add_parser('serve')
    serve.add_argument('--host', default='0.0.0.0')
    serve.add_argument('--port', type=int, default=8000)
    collect = commands.add_parser('collect', add_help=False)
    collect.add_argument('source', choices=['solar-wind', 'geomagnetic'])
    for name in ['refresh', 'aggregate', 'audit', 'cleanup', 'check-health', 'migrate']:
        commands.add_parser(name, add_help=False)
    commands.add_parser('status', help='Show stored collection progress and source freshness')
    schedule = commands.add_parser('schedule')
    schedule.add_argument('job', choices=['refresh', 'aggregate'])
    args, remainder = parser.parse_known_args()
    if args.command in ('serve', 'schedule', 'status') and remainder:
        parser.error('Unrecognized arguments: ' + ' '.join(remainder))
    from common.config import get_config
    get_config()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.command == 'status':
        import asyncio
        import json
        from argus_clio.db.session import get_session_factory, dispose_engine
        from argus_clio.services.collection_status import source_status
        async def read_status():
            try:
                async with get_session_factory()() as session:
                    return await source_status(session)
            finally:
                await dispose_engine()
        print(json.dumps(asyncio.run(read_status()), default=str, indent=2))
        return
    if args.command == 'serve':
        import uvicorn
        uvicorn.run('argus_clio.main:app', host=args.host, port=args.port)
        return
    if args.command == 'migrate':
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
        return
    from argus_clio.commands._runner import run_command
    if args.command == 'collect':
        run_command(lambda: invoke(args.source, remainder))
    elif args.command == 'schedule':
        from argus_clio.scheduler import work
        run_command(lambda: work(args.job, lambda: invoke(args.job)))
    elif args.command in ('refresh', 'aggregate') and not any(arg in ('-h', '--help') for arg in remainder):
        from argus_clio.scheduler import execute
        run_command(lambda: execute(args.command, lambda: invoke(args.command, remainder)))
    else:
        run_command(lambda: invoke(args.command, remainder))
