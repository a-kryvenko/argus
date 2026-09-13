"""Prophet forecasting, publication, CSV export and internal read service."""
import argparse
import importlib
import logging

PRODUCTS = {
    'all': 'generate_forecast',
    'wind': 'generate_wind_forecast',
    'kp': 'generate_kp_forecast',
    'hmf': 'generate_hmf_forecast',
    'density': 'generate_atmospheric_density_forecast',
}


def generate(product: str, trigger='manual', *, scheduled_slot=None) -> None:
    from common.config import get_config
    from argus_prophet.ledger import RunRecorder
    from argus_prophet.observations import load_inputs
    recorder = RunRecorder.begin(product, trigger, get_config(), scheduled_slot=scheduled_slot)
    logging.info('Prophet run %s started (%s)', recorder.run_id, product)
    try:
        inputs = load_inputs()
        recorder.snapshot(inputs)
        command = importlib.import_module(f'argus_prophet.commands.{PRODUCTS[product]}')
        command.main(inputs=inputs, recorder=recorder)
        recorder.finish()
    except BaseException as exc:
        try:
            recorder.finish(error=exc)
        except Exception:
            logging.exception('Could not record failure for run %s', recorder.run_id)
        raise
    else:
        logging.info('Prophet run %s completed', recorder.run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    generate_parser = commands.add_parser('generate', help='Generate forecasts once')
    generate_parser.add_argument('product', nargs='?', choices=PRODUCTS, default='all')
    commands.add_parser('worker', help='Generate hourly at :10 UTC; retry failures')
    commands.add_parser('migrate', add_help=False)
    serve = commands.add_parser('serve', help='Serve published forecast contracts')
    serve.add_argument('--host', default='0.0.0.0')
    serve.add_argument('--port', type=int, default=8000)
    export_parser = commands.add_parser('export', help='Retry pending current CSV exports')
    export_parser.add_argument('--force', action='store_true', help='Restore all current CSVs from published releases')
    commands.add_parser('publish-existing', help='Publish complete recorded runs for the read cutover')
    importer = commands.add_parser('import-schedule', help='Import the previous filesystem completion marker once')
    from pathlib import Path
    importer.add_argument('--marker', type=Path, help='Legacy marker path; defaults to the previous state directory')
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
    from argus_prophet.commands._runner import run_command
    from argus_prophet.worker import generation_lock, work
    from argus_prophet.exports import export_current
    if args.command == 'worker':
        run_command(lambda: work(lambda slot: generate('all', trigger='scheduled', scheduled_slot=slot), export=export_current))
    else:
        def once():
            with generation_lock():
                if args.command == 'import-schedule':
                    from argus_prophet.worker import import_schedule
                    logging.info('Legacy schedule marker imported: %s', import_schedule(args.marker))
                    return
                if args.command == 'publish-existing':
                    from argus_prophet.publication import publish_existing
                    logging.info('Published %s existing product releases', publish_existing())
                    return
                elif args.command == 'generate':
                    generate(args.product)
                # Failure here is an export failure: the committed release survives.
                export_current(force=args.force if args.command == 'export' else False)
        run_command(once)
