"""Clio command parsing and execution; handlers only perform their operation."""
import argparse
import asyncio
import logging
from datetime import UTC, datetime

from clio.ingestion.products import OBSERVATIONS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    serve = commands.add_parser('serve')
    serve.add_argument('--host', default='0.0.0.0')
    serve.add_argument('--port', type=int, default=8000)

    commands.add_parser('collect').add_argument(
        'metrics', nargs='*', help='Configured observations, sdo, donki or enlil (default: all)'
    )
    
    commands.add_parser('normalize')

    commands.add_parser('backfill')

    commands.add_parser('cleanup')

    args = parser.parse_args(argv)

    args.now = datetime.now(UTC)

    if args.command == 'collect':
        if any(m not in OBSERVATIONS and m not in {'sdo', 'donki', 'enlil'} for m in args.metrics):
            parser.error('Unknown observation metric')
        args.metrics = args.metrics or None

    from common.runtime import run_command

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    
    result = run_command(lambda: invoke(args.command, args))
    
    if isinstance(result, dict):
        incomplete = result.get('status') in ('partial', 'failed')
        has_failures = bool(result.get('failed_metrics') or result.get('failed'))
        result = 1 if incomplete or has_failures else 0
        
    if result:
        raise SystemExit(result)


def invoke(name, args):
    if name == 'serve':
        import uvicorn
        uvicorn.run('clio.main:app', host=args.host, port=args.port)
        return
    if name == 'collect':
        from clio.commands.collect import run
    elif name == 'backfill':
        from clio.commands.backfill import run
    elif name == 'normalize':
        from clio.commands.normalize import run
    elif name == 'cleanup':
        from clio.commands.cleanup import run
    else:
        raise ValueError(f'Unknown Clio command: {name}')
    
    from common.db.session import dispose_engine

    async def execute():
        try:
            return await run(args)
        finally:
            await dispose_engine()

    return asyncio.run(execute())


if __name__ == '__main__':
    main()
