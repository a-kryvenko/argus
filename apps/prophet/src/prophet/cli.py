"""Prophet forecasting, publication and internal read service."""
import argparse

from prophet.services.generation.products import GENERATION_CHOICES


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    generate_parser = commands.add_parser('generate', aliases=['refresh'], help='Generate forecasts once')
    generate_parser.add_argument('product', nargs='?', choices=GENERATION_CHOICES, default='all')
    commands.add_parser('migrate', add_help=False)
    serve = commands.add_parser('serve', help='Serve published forecast contracts')
    serve.add_argument('--host', default='0.0.0.0')
    serve.add_argument('--port', type=int, default=8000)
    cleanup = commands.add_parser('cleanup', help='Preview old history cleanup; preserve current releases')
    cleanup.add_argument('--days', type=int, default=90)
    cleanup.add_argument('--apply', action='store_true')
    from prophet.services.generation.products import VERIFIED_PRODUCTS
    for command in ('verify', 'verification-report'):
        verification = commands.add_parser(command, help='Verify published forecasts against later observations')
        verification.add_argument('product', nargs='?', default='solar-wind-speed', choices=(*VERIFIED_PRODUCTS, 'all'))
        verification.add_argument('--days', type=int, choices=range(1, 26), default=7)
    args, remaining = parser.parse_known_args(argv)
    if args.command == 'migrate':
        from prophet.commands.migrate import run
        return run(remaining)
    if remaining:
        parser.error('Unrecognized arguments: ' + ' '.join(remaining))
    if args.command == 'serve':
        import uvicorn
        uvicorn.run('prophet.main:app', host=args.host, port=args.port)
        return
    if args.command in ('generate', 'refresh'):
        from prophet.commands.generation import run
    elif args.command in ('verify', 'verification-report'):
        from prophet.commands.verification import run
    elif args.command == 'cleanup':
        from prophet.commands.cleanup import run
    try:
        return run(args)
    except ValueError as exc:
        parser.error(str(exc))
