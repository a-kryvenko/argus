"""Runs inside a Python HTTP container; never prints credentials or responses."""
import json
import os
import time
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler


def check(options):
    service, base = options['service'], options['base']
    opener = build_opener(ProxyHandler({}))

    def request(path, *, method='GET', body=None, headers=None, expected=200):
        encoded = json.dumps(body).encode() if body is not None else None
        request_headers = {'User-Agent': 'Argus-Deploy/1', **(headers or {})}
        if encoded is not None:
            request_headers['Content-Type'] = 'application/json'
        try:
            with opener.open(Request(base + path, data=encoded, method=method, headers=request_headers), timeout=30) as response:
                if response.status != expected:
                    raise RuntimeError('Unexpected smoke response status')
                return json.load(response), response.headers
        except HTTPError as error:
            if error.code == expected:
                return None, error.headers
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
        data, _ = request('/internal/v1/observations/summary', headers={
            'Authorization': 'Bearer ' + os.environ['OBSERVATIONS_SERVICE_TOKEN']})
        if (not data.get('success') or not isinstance(data.get('data'), dict)
                or not {'solar_wind', 'geomagnetic'} <= data['data'].keys()):
            raise RuntimeError('Invalid observations summary')
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
        origin = os.environ['DASHBOARD_ORIGINS'].split(',')[0].strip()
        request('/dashboard/me', expected=401)
        user, headers = request('/dashboard/login', method='POST', body={
            'username': options['username'], 'password': options['password']}, headers={'Origin': origin})
        cookie = headers.get('Set-Cookie', '').split(';', 1)[0]
        if not cookie.startswith('argus_session='):
            raise RuntimeError('Missing login session')
        session_headers = {'Cookie': cookie, 'Origin': origin}
        try:
            data, _ = request('/dashboard/me', headers=session_headers)
            if not data.get('success') or data['data']['id'] != user['data']['id']:
                raise RuntimeError('Invalid authenticated session')
            for path in ('/public/observations/summary', '/public/forecasts/solar-wind-speed'):
                data, _ = request(path)
                if not data.get('success') or data.get('data') is None:
                    raise RuntimeError('Invalid public API response')
            # Authenticated read only: verifies the API -> Intelligence contract without inference.
            data, _ = request('/dashboard/drag-monitor', headers=session_headers)
            if not data.get('success') or not isinstance(data.get('data'), list):
                raise RuntimeError('Invalid Intelligence API response')
        finally:
            request('/dashboard/logout', method='POST', headers=session_headers)
    print(f'[deploy] {service}: smoke passed', flush=True)
