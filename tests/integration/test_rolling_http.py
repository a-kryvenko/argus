"""Opt-in real Docker test: traffic, in-flight requests and failed cutovers."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.request import urlopen
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/deployment'))
import rolling

pytestmark = pytest.mark.skipif(os.getenv('TEST_ROLLING_DOCKER') != '1', reason='Set TEST_ROLLING_DOCKER=1')

SERVER = '''import json, os, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
active = 0
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global active
        if self.path == '/slow':
            active += 1
            try: time.sleep(3)
            finally: active -= 1
        status = 503 if os.getenv('BAD_SMOKE') == '1' and self.path.endswith('/summary') else 200
        payload = {'version': os.environ['VERSION'], 'active': active}
        if self.path.endswith('/summary'):
            payload = {'success': True, 'data': {'solar_wind': {}, 'geomagnetic': {}}}
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args): pass
ThreadingHTTPServer(('0.0.0.0', int(os.getenv('PORT', '8000'))), Handler).serve_forever()
'''


@pytest.mark.parametrize('failure', [None, 'direct', 'routed'])
@pytest.mark.parametrize('legacy', [False, True])
def test_live_replacement_and_rollback(tmp_path, monkeypatch, failure, legacy):
    project = 'argus-roll-test-' + uuid4().hex[:10]
    (tmp_path / '.env').touch()
    (tmp_path / '.env.local').touch()
    server = tmp_path / 'server.py'
    server.write_text(SERVER)
    deployment = rolling.Deployment.__new__(rolling.Deployment)
    deployment.root, deployment.bundle = tmp_path, ROOT / 'scripts/deployment'
    deployment.base = ['docker', 'compose', '--project-directory', str(tmp_path), '--project-name', project]
    deployment.runtime = tmp_path / 'compose.json'
    deployment.state_path = tmp_path / 'state.json'
    deployment.minimum_available = 0
    deployment.credentials = {'DEPLOY_SMOKE_USERNAME': 'test', 'DEPLOY_SMOKE_PASSWORD': 'unused'}
    spec = {'image': 'python:3.12-alpine', 'working_dir': '/app', 'restart': 'unless-stopped',
            'command': ['sh', '-c', 'mkdir -p .venv/bin && ln -sf /usr/local/bin/python .venv/bin/python && exec python /server.py'],
            'environment': {'VERSION': 'old', 'OBSERVATIONS_SERVICE_TOKEN': 'test'},
            'volumes': [{'type': 'bind', 'source': str(server), 'target': '/server.py', 'read_only': True}],
            'networks': {'observations': {}},
            'healthcheck': {'test': ['CMD', 'python', '-c', "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready')"],
                            'interval': '1s', 'timeout': '2s', 'retries': 5}}
    desired = copy.deepcopy(spec)
    desired['environment']['VERSION'] = 'new'
    if failure == 'direct':
        desired['environment']['BAD_SMOKE'] = '1'
    deployment.desired = {'name': project, 'services': {'clio': desired}, 'networks': {'observations': {}}}
    deployment.state = {'router_version': 2, 'active': {'clio': {'slot': 'blue', 'spec': spec}}}
    if legacy:
        deployment.state.pop('router_version')
    original_router = rolling.router

    def published_router(state, root):
        result = original_router(state, root)
        result['ports'] = [{'target': 8101, 'published': '0', 'host_ip': '127.0.0.1', 'protocol': 'tcp'}]
        return result

    monkeypatch.setattr(rolling, 'router', published_router)
    original_legacy_router = rolling.legacy_router
    def published_legacy_router(service, specification, root):
        result = original_legacy_router(service, specification, root)
        result['ports'] = [{'target': 8000, 'published': '0', 'host_ip': '127.0.0.1', 'protocol': 'tcp'}]
        return result
    monkeypatch.setattr(rolling, 'legacy_router', published_legacy_router)
    deployment.save()
    deployment.route('clio', 'clio-blue')
    deployment.install_runtime()
    stop = threading.Event()
    responses, errors, slow = [], [], []
    thread = slow_thread = None
    try:
        deployment.compose('up', '-d', '--no-deps', '--wait', '--wait-timeout', '90', 'clio-blue', *(['clio'] if legacy else []))
        deployment.prepare_router()
        address = deployment.compose('port', rolling.ROUTER, '8101', capture=True).strip()
        base = 'http://' + address

        legacy_base = 'http://' + deployment.compose('port', 'clio', '8000', capture=True).strip() if legacy else base

        def get(path='/', url=base):
            with urlopen(url + path, timeout=15) as response:
                return json.load(response)

        deadline = time.monotonic() + 15
        while True:
            try:
                assert get()['version'] == 'old'
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.1)

        def traffic():
            while not stop.is_set():
                try:
                    responses.append(get()['version'])
                except Exception as error:
                    errors.append(str(error))
                stop.wait(.02)

        thread = threading.Thread(target=traffic)
        thread.start()
        slow_thread = threading.Thread(target=lambda: slow.append(get('/slow', legacy_base)))
        slow_thread.start()
        deadline = time.monotonic() + 10
        while get()['active'] == 0:
            assert time.monotonic() < deadline
            time.sleep(.02)
        original_smoke = deployment.smoke

        def smoke(service, slot, routed=False):
            original_smoke(service, slot, routed)
            if failure == 'routed' and slot == 'clio-green' and routed:
                raise RuntimeError('injected routed failure')

        deployment.smoke = smoke
        if failure:
            with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
                deployment.replace('clio')
        else:
            deployment.replace('clio')
        slow_thread.join(timeout=15)
        assert slow == [{'version': 'old', 'active': 0}]
        assert get()['version'] == ('old' if failure else 'new')
        assert deployment.state['active']['clio']['slot'] == ('blue' if failure else 'green')
        assert 'pending' not in deployment.state
        assert errors == []
        assert responses
        if legacy:
            assert get(url=legacy_base)['version'] == ('old' if failure else 'new')
        if not failure:
            deployment.retire_legacy_routers()
            assert deployment.worker_containers('clio') == []
            assert get()['version'] == 'new'
    finally:
        stop.set()
        if thread:
            thread.join(timeout=20)
        if slow_thread:
            slow_thread.join(timeout=20)
        deployment.compose('down', '--volumes', '--remove-orphans')


def test_shared_router_keeps_listeners_and_networks_independent(tmp_path):
    project = 'argus-router-test-' + uuid4().hex[:10]
    server = tmp_path / 'server.py'
    server.write_text(SERVER)
    deployment = rolling.Deployment.__new__(rolling.Deployment)
    deployment.root, deployment.bundle = tmp_path, ROOT / 'scripts/deployment'
    deployment.base = ['docker', 'compose', '--project-directory', str(tmp_path), '--project-name', project]
    deployment.runtime, deployment.state_path = tmp_path / 'compose.json', tmp_path / 'state.json'
    deployment.minimum_available = 0
    deployment.credentials = {'DEPLOY_SMOKE_USERNAME': 'test', 'DEPLOY_SMOKE_PASSWORD': 'unused'}
    specs = {}
    for service, networks in [('clio', ['observations']), ('api', ['application', 'observations']),
                              ('frontend', ['application'])]:
        specs[service] = {
            'image': 'python:3.12-alpine', 'working_dir': '/app',
            'command': ['sh', '-c', 'mkdir -p .venv/bin && ln -sf /usr/local/bin/python .venv/bin/python && exec python /server.py'],
            'environment': {'VERSION': service + '-old', 'PORT': str(rolling.HTTP[service]),
                            'OBSERVATIONS_SERVICE_TOKEN': 'test'},
            'volumes': [{'type': 'bind', 'source': str(server), 'target': '/server.py', 'read_only': True}],
            'networks': {name: {} for name in networks},
            'healthcheck': {'test': ['CMD', 'python', '-c',
                                    f"import urllib.request; urllib.request.urlopen('http://127.0.0.1:{rolling.HTTP[service]}/')"],
                            'interval': '1s', 'timeout': '2s', 'retries': 5},
        }
    deployment.state = {'router_version': 2, 'active': {
        name: {'slot': 'blue', 'spec': copy.deepcopy(spec)} for name, spec in specs.items()}}
    deployment.desired = {'name': project, 'services': specs,
                          'networks': {'application': {}, 'observations': {}}}
    deployment.desired['services']['clio']['environment']['VERSION'] = 'clio-new'
    deployment.save()
    deployment.install_runtime()
    try:
        deployment.compose('up', '-d', '--no-deps', '--wait', '--wait-timeout', '90',
                           'clio-blue', 'api-blue', 'frontend-blue')
        deployment.prepare_router()

        def get(client, url):
            source = "import json,sys,urllib.request; print(json.load(urllib.request.urlopen(sys.argv[1], timeout=3))['version'])"
            return deployment.compose('exec', '-T', client, 'python', '-c', source, url, capture=True).strip()

        for service in specs:
            assert get('api-blue', rolling.routed_url(service)) == service + '-old'
        assert get('frontend-blue', rolling.routed_url('api')) == 'api-old'
        # A public-network client cannot reach the observations listener through
        # the router's public-facing address, even though it is one container.
        with pytest.raises(subprocess.CalledProcessError):
            get('frontend-blue', 'http://application-router:8101')
        deployment.replace('clio')
        assert get('api-blue', rolling.routed_url('clio')) == 'clio-new'
        for service in ('api', 'frontend'):
            assert get('api-blue', rolling.routed_url(service)) == service + '-old'
        active = deployment.compose('ps', '--services', '--status', 'running', capture=True).split()
        assert set(active) == {rolling.ROUTER, 'clio-green', 'api-blue', 'frontend-blue'}
    finally:
        deployment.compose('down', '--volumes', '--remove-orphans')
