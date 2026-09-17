"""Validate Prophet releases and manage Intelligence processing and migrations."""
import argparse
from datetime import UTC, datetime
import json
import os
from uuid import UUID

import httpx
from common.schemas.forecast_release import ForecastRelease, PRODUCT_ARTIFACTS

MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def check(product, *, release_id=None, transport=None):
    if product not in PRODUCT_ARTIFACTS:
        raise ValueError('Unsupported product')
    url, token = os.getenv('FORECASTS_URL'), os.getenv('FORECASTS_SERVICE_TOKEN')
    if not url or not token:
        raise ValueError('FORECASTS_URL and FORECASTS_SERVICE_TOKEN are required')
    suffix = 'latest' if release_id is None else f'releases/{UUID(str(release_id))}'
    with httpx.Client(timeout=httpx.Timeout(60, connect=10), follow_redirects=False,
                      trust_env=False, transport=transport) as client:
        with client.stream('GET', url.rstrip('/') + f'/internal/v1/forecasts/{product}/{suffix}',
                           headers={'Authorization': f'Bearer {token}'}) as response:
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_bytes():
                content.extend(chunk)
                if len(content) > MAX_RESPONSE_BYTES:
                    raise ValueError('Response too large')
    release = ForecastRelease.model_validate_json(content)
    if release.product != product or (release_id is not None and str(release.release_id) != str(release_id)):
        raise ValueError('Unexpected release')
    return {'contract_version': 1, 'service': 'intelligence', 'status': 'ok', 'mode': 'stub',
            'checked_at': datetime.now(UTC).isoformat(), 'product': product,
            'release_id': str(release.release_id), 'run_id': str(release.run_id),
            'issue_time': release.issue_time.isoformat(), 'artifact_count': len(release.artifacts),
            'risk_assessment': None,
            'note': 'HTTP integration and release contract validated; model readiness and satellite risks are not assessed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    checker = commands.add_parser('check')
    checker.add_argument('product', choices=PRODUCT_ARTIFACTS, nargs='?', default='solar-wind-speed')
    checker.add_argument('--release-id', type=UUID)
    for command in ('worker', 'process', 'status'):
        sub = commands.add_parser(command)
        sub.add_argument('product', choices=PRODUCT_ARTIFACTS, nargs='?', default='solar-wind-speed')
    commands.add_parser('migrate', add_help=False)
    args, remaining = parser.parse_known_args()
    if args.command == 'migrate':
        from pathlib import Path
        from alembic.config import Config, CommandLine
        runner = CommandLine(prog='intelligence migrate')
        if not remaining:
            runner.parser.print_help()
            return
        options = runner.parser.parse_args(remaining)
        config = Config()
        config.cmd_opts = options
        config.set_main_option('script_location', str(Path(__file__).parent / 'migrations'))
        runner.run_cmd(config, options)
        return
    if remaining:
        parser.error('Unrecognized arguments: ' + ' '.join(remaining))
    if args.command in ('worker', 'process', 'status'):
        from argus_intelligence import worker
        try:
            if args.command == 'worker':
                worker.worker(args.product)
                return
            result = worker.status(args.product) if args.command == 'status' else worker.process_once(args.product)
        except Exception:
            print(json.dumps({'service': 'intelligence', 'status': 'error', 'error': 'Database or release processing unavailable'}))
            raise SystemExit(1) from None
        print(json.dumps(result, default=str))
        return
    try:
        result = check(args.product, release_id=args.release_id)
    except (httpx.HTTPError, ValueError):
        # Never log response bodies, credentials or validation input payloads.
        print(json.dumps({'service': 'intelligence', 'status': 'error', 'mode': 'stub',
                          'product': args.product, 'error': 'Forecast configuration, HTTP request or release validation failed.'}))
        raise SystemExit(1) from None
    print(json.dumps(result))
