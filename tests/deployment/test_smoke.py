"""Deployment smoke checks against the current Clio response contract."""
import importlib.util
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

spec = importlib.util.spec_from_file_location(
    'smoke', Path(__file__).resolve().parents[2] / 'scripts/deployment/smoke.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


@pytest.mark.parametrize('payload,error', [
    ({'success': True, 'data': {'points': []}}, None),
    ({'success': True, 'data': {'solar_wind': {}, 'geomagnetic': {}}}, 'Invalid latest observations'),
    ({'success': False, 'data': {'points': []}}, 'Invalid latest observations'),
    (None, 'returned 503'),
])
def test_clio_smoke_uses_latest_contract(monkeypatch, payload, error):
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            if request.full_url.endswith('/health/ready'):
                data = {}
            else:
                assert request.full_url == 'http://clio/internal/v1/observations/latest'
                assert request.get_header('Authorization') == 'Bearer test-token'
                if payload is None:
                    raise HTTPError(request.full_url, 503, 'Unavailable', {}, None)
                data = payload
            response = io.BytesIO(json.dumps(data).encode())
            response.status = 200
            response.headers = {}
            return response

    monkeypatch.setattr(smoke, 'build_opener', lambda *args: Opener())
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'test-token')
    options = {'service': 'clio', 'base': 'http://clio'}
    if error:
        with pytest.raises(RuntimeError, match=error):
            smoke.check(options)
    else:
        smoke.check(options)
    assert len(calls) == 2
