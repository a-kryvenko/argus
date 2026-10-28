"""Prophet command parsing and execution; handlers only perform their operation."""
import argparse
import asyncio
import logging

from prophet.services.generation.products import GENERATION_CHOICES, VERIFIED_PRODUCTS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    serve = commands.add_parser('serve')
    serve.add_argument('--host', default='0.0.0.0')
    serve.add_argument('--port', type=int, default=8000)

    generation = commands.add_parser('generation')
    generation.add_argument('product', nargs='?', choices=GENERATION_CHOICES, default='all')

    verification = commands.add_parser('verification')
    verification.add_argument('product', nargs='?', choices=(*VERIFIED_PRODUCTS, 'all'), default='solar-wind-speed')

    cleanup = commands.add_parser('cleanup')
    cleanup.add_argument('--apply', action='store_true')

    args = parser.parse_args(argv)

    from common.runtime import run_command

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    result = run_command(lambda: invoke(args.command, args))
    if result:
        raise SystemExit(result)


def invoke(name, args):
    if name == 'serve':
        import uvicorn
        uvicorn.run('prophet.main:app', host=args.host, port=args.port)
        return
    if name == 'generation':
        from prophet.commands.generation import run
    elif name == 'verification':
        from prophet.commands.verification import run
    elif name == 'cleanup':
        from prophet.commands.cleanup import run
    else:
        raise ValueError(f'Unknown Prophet command: {name}')

    from common.db.session import dispose_engine

    async def execute():
        try:
            return await run(args)
        finally:
            await dispose_engine()

    return asyncio.run(execute())


if __name__ == '__main__':
    main()
