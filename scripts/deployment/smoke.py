"""Read-only HTTP smoke checks executed inside Python service containers."""
import json
import os
import time
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler


def check(options):
    service, base = options['service'], options['base']
    opener = build_opener(ProxyHandler({}))

    def request(path, *, headers=None):
        request_headers = {'User-Agent': 'Argus-Deploy/1', **(headers or {})}
        try:
            with opener.open(Request(base + path, headers=request_headers), timeout=30) as response:
                if response.status != 200:
                    raise RuntimeError('Unexpected smoke response status')
                return json.load(response), response.headers
        except HTTPError as error:
            raise RuntimeError(f'{service}: smoke endpoint {path} returned {error.code}') from None

    deadline = time.monotonic() + 5
    while True:
        _, headers = request('/health/ready')
        if not options.get('expected_upstream') or headers.get('X-Argus-Upstream') == options['expected_upstream']:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('Router did not select the expected application instance')
        time.sleep(.1)
    if service == 'clio':
        data, _ = request('/internal/v1/observations/latest', headers={
            'Authorization': 'Bearer ' + os.environ['OBSERVATIONS_SERVICE_TOKEN']})
        if (not data.get('success') or not isinstance(data.get('data'), dict)
                or not isinstance(data['data'].get('points'), list)):
            raise RuntimeError('Invalid latest observations')
    elif service == 'prophet-api':
        from common.schemas.forecast_release import ForecastRelease
        data, _ = request('/internal/v1/forecasts/solar-wind-speed/latest', headers={
            'Authorization': 'Bearer ' + os.environ['FORECASTS_SERVICE_TOKEN']})
        release = ForecastRelease.model_validate(data)
        if release.product != 'solar-wind-speed' or not release.artifacts:
            raise RuntimeError('Invalid forecast release')
    elif service == 'intelligence-api':
        data, _ = request('/internal/v1/drag-monitor/0', headers={
            'Authorization': 'Bearer ' + os.environ['INTELLIGENCE_SERVICE_TOKEN']})
        if not isinstance(data, list):
            raise RuntimeError('Invalid Intelligence profile list')
    elif service == 'api':
        for path in ('/observations', '/forecasts/solar-wind-speed'):
            data, _ = request(path)
            if not data.get('success') or data.get('data') is None:
                raise RuntimeError('Invalid public API response')
    print(f'[deploy] {service}: smoke passed', flush=True)
