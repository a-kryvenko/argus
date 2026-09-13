"""Installed production entry point: prophet generate [product] / prophet worker."""
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


def generate(product: str) -> None:
    command = importlib.import_module(f'argus_prophet.commands.{PRODUCTS[product]}')
    command.main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    generate_parser = commands.add_parser('generate', help='Generate forecasts once')
    generate_parser.add_argument('product', nargs='?', choices=PRODUCTS, default='all')
    commands.add_parser('worker', help='Generate hourly at :10 UTC; retry failures')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    from argus_prophet.commands._runner import run_command
    from argus_prophet.worker import generation_lock, work
    if args.command == 'worker':
        run_command(lambda: work(lambda: generate('all')))
    else:
        def once():
            with generation_lock():
                generate(args.product)
        run_command(once)
