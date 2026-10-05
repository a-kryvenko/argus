"""Prophet forecasting, publication and internal read service."""
import argparse
from importlib import import_module

from argus_prophet.services.generation.products import GENERATION_CHOICES


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo', help='Download inputs, calculate and publish the isolated demo dataset')
    demo.add_argument('scenario', nargs='?', default='january-2026')
    demo.add_argument('--refresh-data', action='store_true', help='Refresh cached observation archives')
    generate_parser = commands.add_parser('generate', aliases=['refresh'], help='Generate forecasts once')
    generate_parser.add_argument('product', nargs='?', choices=GENERATION_CHOICES, default='all')
    commands.add_parser('worker', help='Generate hourly at :10 UTC; retry failures')
    commands.add_parser('migrate', add_help=False)
    serve = commands.add_parser('serve', help='Serve published forecast contracts')
    serve.add_argument('--host', default='0.0.0.0')
    serve.add_argument('--port', type=int, default=8000)
    slots = commands.add_parser('slots', help='List product schedule slots and attempt counts')
    slots.add_argument('--limit', type=int, default=20)
    cleanup = commands.add_parser('cleanup', help='Preview old history cleanup; preserve current releases')
    cleanup.add_argument('--days', type=int, default=90)
    cleanup.add_argument('--apply', action='store_true')
    status_parser = commands.add_parser('status', help='Inspect current release age and input diagnostics')
    status_parser.add_argument('product', nargs='?', default='all', choices=GENERATION_CHOICES)
    from argus_prophet.services.generation.products import VERIFIED_PRODUCTS
    for command in ('verify', 'verification-report'):
        verification = commands.add_parser(command, help='Verify published forecasts against later observations')
        verification.add_argument('product', nargs='?', default='solar-wind-speed', choices=(*VERIFIED_PRODUCTS, 'all'))
        verification.add_argument('--days', type=int, choices=range(1, 26), default=7)
    runs = commands.add_parser('runs', help='List recorded executions')
    runs.add_argument('--limit', type=int, default=20)
    show = commands.add_parser('show-run', help='Show execution evidence')
    from uuid import UUID
    show.add_argument('run_id', type=UUID)
    show.add_argument('--inputs', action='store_true', help='Include the saved observation snapshot')
    args, remaining = parser.parse_known_args(argv)
    if args.command == 'migrate':
        from argus_prophet.commands.migrate import run
        return run(remaining)
    if remaining:
        parser.error('Unrecognized arguments: ' + ' '.join(remaining))
    if args.command == 'serve':
        import uvicorn
        uvicorn.run('argus_prophet.main:app', host=args.host, port=args.port)
        return
    handlers = {
        'demo': 'demo',
        'generate': 'generation', 'refresh': 'generation', 'worker': 'generation',
        'verify': 'verification', 'verification-report': 'verification',
        'runs': 'inspection', 'show-run': 'inspection', 'slots': 'inspection', 'status': 'inspection',
        'cleanup': 'cleanup',
    }
    handler = import_module('argus_prophet.commands.' + handlers[args.command])
    try:
        return handler.run(args)
    except ValueError as exc:
        parser.error(str(exc))
