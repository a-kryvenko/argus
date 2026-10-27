"""Clio command parsing and execution; handlers only perform their operation."""
import argparse
import logging
from datetime import UTC, datetime, timedelta

from clio.commands._arguments import boundary
from clio.commands.runner import invoke
from clio.ingestion.products import OBSERVATIONS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    serve = commands.add_parser('serve')
    serve.add_argument('--host', default='0.0.0.0')
    serve.add_argument('--port', type=int, default=8000)

    commands.add_parser('collect').add_argument(
        'metrics', nargs='*', help='Configured observations; sdo selects AIA/HMI images (default: all)')
    commands.add_parser('normalize')
    commands.add_parser('collect-events', help='Collect and version DONKI CME/ENLIL/HSS/IPS responses')
    commands.add_parser('collect-enlil', help='Collect and version compact NOAA ENLIL L1 profiles')

    backfill = commands.add_parser('backfill')
    backfill.add_argument('--from', dest='start', type=boundary)
    backfill.add_argument('--to', dest='end', type=boundary)
    backfill.add_argument('metrics', nargs='*',
                          help='Fill configured gaps; sdo selects retained AIA/HMI images (default: all)')

    backfill.add_argument('--batch', action='store_true', help='Bound SDO warmup to one configured batch')
    commands.add_parser('sdo-cleanup')
    commands.add_parser('migrate', add_help=False)

    args, remainder = parser.parse_known_args(argv)

    if args.command != 'migrate' and remainder:
        parser.error('Unrecognized arguments: ' + ' '.join(remainder))

    args.now = datetime.now(UTC)

    if args.command in ('collect', 'backfill'):
        if any(m not in OBSERVATIONS and m != 'sdo' for m in args.metrics):
            parser.error('Unknown observation metric')
        args.metrics = args.metrics or None

    if args.command == 'backfill':
        if (args.start is None) != (args.end is None):
            parser.error('--from and --to must be specified together')

    if args.command == 'backfill' and args.start is not None:
        if not timedelta(0) < args.end - args.start <= timedelta(days=31) or args.end > args.now:
            parser.error('Choose a past range of at most 31 days, with --from before --to')

    from common.runtime import run_command
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    result = run_command(lambda: dispatch(args, remainder))
    if result:
        raise SystemExit(result)


def dispatch(args, remainder):
    if args.command != 'migrate':
        from clio.config import load_observation_config
        load_observation_config()
    if args.command == 'serve':
        import uvicorn
        uvicorn.run('clio.main:app', host=args.host, port=args.port)
    elif args.command == 'migrate':
        from clio.commands.migrate import run
        return run(remainder)
    else:
        result = invoke(args.command, args)
        if isinstance(result, dict):
            return int(bool(result.get('failed_metrics') or result.get('failed')))
        return result


if __name__ == '__main__':
    main()
