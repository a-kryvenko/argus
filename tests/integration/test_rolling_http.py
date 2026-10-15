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
ThreadingHTTPServer(('0.0.0.0', 8000), Handler).serve_forever()
'''


@pytest.mark.parametrize('failure', [None, 'direct', 'routed'])
def test_live_replacement_and_rollback(tmp_path, monkeypatch, failure):
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
    deployment.state = {'active': {'clio': {'slot': 'blue', 'spec': spec}}}
    original_router = rolling.router

    def published_router(service, specification, root):
        result = original_router(service, specification, root)
        result['ports'] = [{'target': 8000, 'published': '0', 'host_ip': '127.0.0.1', 'protocol': 'tcp'}]
        return result

    monkeypatch.setattr(rolling, 'router', published_router)
    deployment.save()
    deployment.route('clio', 'clio-blue')
    deployment.install_runtime()
    stop = threading.Event()
    responses, errors, slow = [], [], []
    thread = slow_thread = None
    try:
        deployment.compose('up', '-d', '--no-deps', '--wait', '--wait-timeout', '90', 'clio-blue', 'clio')
        address = deployment.compose('port', 'clio', '8000', capture=True).strip()
        base = 'http://' + address

        def get(path='/'):
            with urlopen(base + path, timeout=15) as response:
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
        slow_thread = threading.Thread(target=lambda: slow.append(get('/slow')))
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
    finally:
        stop.set()
        if thread:
            thread.join(timeout=20)
        if slow_thread:
            slow_thread.join(timeout=20)
        deployment.compose('down', '--volumes', '--remove-orphans')
