"""Opt-in Docker test for static slots, live traffic and failed cutovers."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/deployment'))
from _rollout.applications import Applications
from _rollout.compose import Compose
from _rollout.database import Database
from _rollout.deployment import Deployment
from _rollout.proxy import NginxProxy
from _rollout.release import Release
from _rollout.applications import HTTP, routed_url, slot_names
from _rollout.proxy import ROUTER
from _rollout.state import DeploymentState

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
        body = json.dumps({'version': os.environ['VERSION'], 'active': active}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args): pass
ThreadingHTTPServer(('0.0.0.0', int(os.environ['PORT'])), Handler).serve_forever()
'''


def install_argus(root):
    (root / 'docker-compose.prod.yml').write_text('{}')
    (root / 'scripts').mkdir()
    shutil.copy2(ROOT / 'argus', root / 'argus')
    shutil.copy2(ROOT / 'scripts/run', root / 'scripts/run')
    (root / '.argus-mode').write_text('prod\n')
    for name in ('.env', '.env.local'):
        (root / name).touch()


def deployment_with_static_slots(root):
    config = json.loads(subprocess.check_output([
        'docker', 'compose', '--env-file', '/dev/null', '--profile', '*',
        '-f', str(ROOT / 'docker-compose.yml'), '-f', str(ROOT / '.deploy/docker-compose.yml'),
        'config', '--no-interpolate', '--format', 'json',
    ], text=True))
    router = config['services'][ROUTER]
    services = {ROUTER: router}
    router['volumes'] = [{'type': 'bind', 'source': str(root / 'routes/internal'),
                          'target': '/etc/nginx/conf.d', 'read_only': True}]
    server = root / 'server.py'
    server.write_text(SERVER)
    for color in ('blue', 'green'):
        for service, name in zip(HTTP, slot_names(color)):
            spec = copy.deepcopy(config['services'][name])
            spec.update(image='python:3.12-alpine', command=['python', '/server.py'],
                        environment={'VERSION': name, 'PORT': str(HTTP[service].port)},
                        volumes=[{'type': 'bind', 'source': str(server), 'target': '/server.py', 'read_only': True}],
                        healthcheck={'test': ['CMD', 'python', '-c',
                            f"import urllib.request; urllib.request.urlopen('http://127.0.0.1:{HTTP[service].port}/')"],
                            'interval': '1s', 'timeout': '2s', 'retries': 20})
            spec['stop_grace_period'] = '1s'
            services[name] = spec
    (root / 'docker-compose.yml').write_text(json.dumps({
        'name': 'argus-color-test-' + uuid4().hex[:10], 'services': services,
        'networks': {network: {} for network in config['networks']},
    }))
    install_argus(root)
    (root / '.active-color').write_text('blue\n')
    compose = Compose(root)
    applications = Applications(root, compose)
    deployment = Deployment(
        compose=compose,
        release=Release(root, root, compose.execute),
        database=Database(root, compose),
        applications=applications,
        proxy=NginxProxy(root, compose, applications),
        state=DeploymentState(root),
    )
    deployment.color = 'blue'
    deployment.applications.publish_static = lambda slot: None
    return deployment


@pytest.mark.parametrize('failure', [None, 'direct', 'routed'])
def test_static_slots_switch_as_one_group_and_preserve_inflight_requests(tmp_path, failure):
    deployment = deployment_with_static_slots(tmp_path)
    stop = threading.Event()
    responses, errors, slow = [], [], []
    thread = slow_thread = None
    def get_from_container(client, url):
        source = "import urllib.request,sys; print(urllib.request.urlopen(sys.argv[1], timeout=5).read().decode())"
        return json.loads(deployment.compose.run('exec', '-T', client, 'python', '-c', source, url, capture=True))
    def smoke(service, slot, routed=False):
        url = routed_url(service) if routed else f'http://127.0.0.1:{HTTP[service].port}'
        assert get_from_container(slot, url)['version'] == slot
        if slot == 'api-green' and ((failure == 'direct' and not routed) or (failure == 'routed' and routed)):
            raise RuntimeError('injected smoke failure')
    deployment.applications.check = smoke
    try:
        deployment.applications.start('blue')
        deployment.proxy.prepare('blue')
        def get(path='/'):
            return json.loads(deployment.compose.run('exec', '-T', ROUTER, 'wget', '-qO-',
                              '-T', '15', '-t', '1', 'http://observations-router:8101' + path, capture=True))
        assert get()['version'] == 'clio-blue'
        with pytest.raises(subprocess.CalledProcessError):
            get_from_container('frontend-blue', 'http://application-router:8101')
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
        if failure:
            with pytest.raises(RuntimeError, match='injected'):
                deployment.swap_blue_green_services()
        else:
            deployment.swap_blue_green_services()
        slow_thread.join(timeout=15)
        assert slow == [{'version': 'clio-blue', 'active': 0}]
        color = 'blue' if failure else 'green'
        assert deployment.state.read_color() == color
        assert get()['version'] == 'clio-' + color
        running = set(deployment.compose.run('ps', '--status', 'running', '--services', capture=True).split())
        assert running == {ROUTER, *slot_names(color)}
        assert responses and not errors
        for service in HTTP:
            assert get_from_container('api-' + color, routed_url(service))['version'] == service + '-' + color
    finally:
        stop.set()
        for active_thread in (thread, slow_thread):
            if active_thread:
                active_thread.join(timeout=20)
        deployment.compose.run('down', '--timeout', '1', '--volumes', '--remove-orphans')


def test_finish_release_reloads_nginx_without_recreating_it_when_compose_changes(tmp_path, monkeypatch):
    configuration = tmp_path / 'nginx.conf'
    configuration.write_text('server { listen 80; location / { return 200 "before"; } }\n')
    service = {'image': 'nginx:stable-alpine', 'environment': {'GENERATION': 'before'},
               'volumes': [str(configuration) + ':/etc/nginx/conf.d/default.conf:ro'],
               'healthcheck': {'test': ['CMD', 'nginx', '-t'], 'interval': '1s', 'retries': 5}}
    config = {'name': 'argus-nginx-test-' + uuid4().hex[:10], 'services': {'nginx': service}}
    path = tmp_path / 'docker-compose.yml'
    path.write_text(json.dumps(config))
    install_argus(tmp_path)
    compose = Compose(tmp_path)
    deployment = Deployment(
        compose=compose, release=SimpleNamespace(install_crontab=lambda: None),
        database=None, applications=None, proxy=NginxProxy(tmp_path, compose, None),
        state=SimpleNamespace(record_release=lambda manifest: None),
    )
    deployment.manifest = {}
    run = compose.run
    def nginx_only(*args, **kwargs):
        if args[-1] in ('proswin', 'alloy'):
            return ''
        return run(*args, **kwargs)
    try:
        run('up', '-d', '--pull', 'never', '--wait', 'nginx')
        container = run('ps', '-q', 'nginx', capture=True)
        service['environment']['GENERATION'] = 'after'
        path.write_text(json.dumps(config))
        configuration.write_text('server { listen 80; location / { return 200 "after"; } }\n')
        monkeypatch.setattr(compose, 'run', nginx_only)
        deployment.finish_release()
        assert run('ps', '-q', 'nginx', capture=True) == container
        assert run('exec', '-T', 'nginx', 'wget', '-qO-', 'http://127.0.0.1/', capture=True) == 'after'
    finally:
        run('down', '--timeout', '1', '--volumes', '--remove-orphans')
