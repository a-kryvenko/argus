"""Prophet forecasting, publication and internal read service."""
import argparse
import logging

from argus_prophet.products import GENERATION_CHOICES, select_products


class GenerationFailed(RuntimeError):
    def __init__(self, failures):
        self.failures = failures
        super().__init__('Forecast generation failed for: ' + ', '.join(failures))


def generate(product: str, trigger='manual', *, scheduled_slot=None) -> None:
    generate_products(select_products(product), trigger, scheduled_slot=scheduled_slot)


def generate_products(products, trigger='manual', *, scheduled_slot=None) -> None:
    from argus_prophet.products import PRODUCTS
    from argus_prophet.generation import calculate
    from common.config import get_config
    from argus_prophet.ledger import RunRecorder, provenance
    from argus_prophet.observations import load_inputs
    if not products or len(set(products)) != len(products) or any(name not in PRODUCTS for name in products):
        raise ValueError('Expected distinct supported forecast products')
    config = get_config()
    details = provenance(config)
    inputs = None
    input_error = None
    failures = {}
    for product in products:
        recorder = RunRecorder.begin(product, trigger, config, scheduled_slot=scheduled_slot,
                                     details=details)
        logging.info('Prophet run %s started (%s)', recorder.run_id, product)
        try:
            if inputs is None and input_error is None:
                try:
                    inputs = load_inputs()
                except Exception as exc:
                    input_error = exc
            if input_error is not None:
                raise input_error
            recorder.snapshot(inputs)
            calculate(product, inputs=inputs, recorder=recorder)
            recorder.finish()
        except BaseException as exc:
            # If failure recording also fails (e.g. lost lock/DB connection), stop.
            # A new writer must recover this attempt before any more work starts.
            recorder.finish(error=exc)
            if not isinstance(exc, Exception):
                raise
            failures[product] = exc
            logging.exception('Prophet run %s failed (%s)', recorder.run_id, product)
        else:
            logging.info('Prophet run %s published (%s)', recorder.run_id, product)
    if failures:
        raise GenerationFailed(failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    generate_parser = commands.add_parser('generate', help='Generate forecasts once')
    generate_parser.add_argument('product', nargs='?', choices=GENERATION_CHOICES, default='all')
    commands.add_parser('worker', help='Generate hourly at :10 UTC; retry failures')
    commands.add_parser('migrate', add_help=False)
    serve = commands.add_parser('serve', help='Serve published forecast contracts')
    serve.add_argument('--host', default='0.0.0.0')
    serve.add_argument('--port', type=int, default=8000)
    slots = commands.add_parser('slots', help='List hourly slots and attempt counts')
    slots.add_argument('--limit', type=int, default=20)
    from common.schemas.forecast_release import PRODUCT_ARTIFACTS
    status_parser = commands.add_parser('status', help='Inspect current release age and input diagnostics')
    status_parser.add_argument('product', choices=PRODUCT_ARTIFACTS)
    runs = commands.add_parser('runs', help='List recorded executions')
    runs.add_argument('--limit', type=int, default=20)
    show = commands.add_parser('show-run', help='Show execution evidence')
    from uuid import UUID
    show.add_argument('run_id', type=UUID)
    show.add_argument('--inputs', action='store_true', help='Include the saved observation snapshot')
    args, remaining = parser.parse_known_args()
    if args.command == 'migrate':
        from pathlib import Path
        from common.config import get_config
        from alembic.config import Config, CommandLine
        get_config()
        cli = CommandLine(prog='prophet migrate')
        if not remaining:
            cli.parser.print_help()
            return
        options = cli.parser.parse_args(remaining)
        config = Config()
        config.cmd_opts = options
        config.set_main_option('script_location', str(Path(__file__).parent / 'migrations'))
        cli.run_cmd(config, options)
        return
    if remaining:
        parser.error('Unrecognized arguments: ' + ' '.join(remaining))
    if args.command == 'serve':
        import uvicorn
        uvicorn.run('argus_prophet.main:app', host=args.host, port=args.port)
        return
    if args.command in ('runs', 'show-run', 'slots', 'status'):
        import json
        from common.config import get_config
        from argus_prophet.ledger import list_runs, describe_run
        get_config()
        try:
            if args.command == 'status':
                from argus_prophet.readiness import product_status
                result = product_status(args.product).model_dump(mode='json')
            elif args.command == 'slots':
                from argus_prophet.worker import list_slots
                result = list_slots(args.limit)
            else:
                result = list_runs(args.limit) if args.command == 'runs' else describe_run(args.run_id, args.inputs)
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(result, default=str, indent=2))
        return
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    from argus_prophet.runtime import run_command
    from argus_prophet.worker import generation_lock, work
    if args.command == 'worker':
        run_command(lambda: work(lambda slot, products: generate_products(
            products, trigger='scheduled', scheduled_slot=slot)))
    else:
        def once():
            with generation_lock():
                generate(args.product)
        run_command(once)
