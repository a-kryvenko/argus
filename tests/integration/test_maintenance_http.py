"""Opt-in check of the production maintenance page using a local nginx image."""
import copy
import json
import os
from pathlib import Path
import subprocess
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

import pytest

from tests.deployment.compose import load_compose

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.getenv('TEST_MAINTENANCE_DOCKER') != '1',
                                reason='Set TEST_MAINTENANCE_DOCKER=1')


def test_page_returns_503_for_all_paths_and_methods_and_starts_only_explicitly(tmp_path):
    service = copy.deepcopy(load_compose('.deploy/docker-compose.yml')['services']['maintenance'])
    service['networks'] = ['default']
    service['environment'] = {}
    service['ports'] = ['127.0.0.1::80']
    service['volumes'] = [mount.replace('./.docker/', str(ROOT / '.deploy/.docker') + '/')
                          for mount in service['volumes']]
    config = tmp_path / 'compose.json'
    config.write_text(json.dumps({'name': 'argus-maintenance-test-' + uuid4().hex[:10],
                                 'services': {'maintenance': service}}))
    def compose(*args):
        return subprocess.check_output(['docker', 'compose', '-f', str(config), *args],
                                       text=True, timeout=60).strip()
    try:
        compose('up', '-d', '--pull', 'never', 'maintenance')
        assert not compose('ps', '-q', 'maintenance')
        compose('up', '-d', '--pull', 'never', '--scale', 'maintenance=1', '--wait', 'maintenance')
        address = compose('port', 'maintenance', '80')
        for method in ('GET', 'POST', 'HEAD'):
            for path in ('/', '/api/v1/observations', '/_next/static/test.js', '/maintenance.html'):
                with pytest.raises(HTTPError) as response:
                    urlopen(Request('http://' + address + path, method=method), timeout=5)
                with response.value as error:
                    assert error.code == 503
                    assert error.headers['Retry-After'] == '60'
                    assert error.headers['Cache-Control'] == 'no-store'
                    if method != 'HEAD':
                        assert b"We'll be back soon" in error.read()
        compose('stop', 'maintenance')
        assert not compose('ps', '-q', 'maintenance')
    finally:
        compose('down', '--volumes', '--remove-orphans')
