"""Exercise the shipped nginx login policy with disposable HTTP upstreams.

TEST_NGINX_IMAGE=nginx:stable-alpine tests/runtime/.venv/bin/python -m pytest -q tests/integration/test_login_edge.py
"""
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def edge(tmp_path_factory):
    image = os.getenv('TEST_NGINX_IMAGE')
    if not image:
        pytest.skip('Set TEST_NGINX_IMAGE to run the disposable nginx integration checks')
    directory = tmp_path_factory.mktemp('login-edge')
    (directory / 'log').mkdir()
    template = (ROOT / '.deploy/nginx/templates/http.conf.template').read_text()
    (directory / 'site.conf').write_text(template.replace('${APP_NAME}', 'dashboard.test'))
    (directory / 'default.conf').write_text((ROOT / '.deploy/nginx/default.conf').read_text())
    # Port 8081 simulates the trusted edge on loopback. Port 80 is an untrusted
    # direct connection. Test-only headers never enter the shipped configuration.
    (directory / 'nginx.conf').write_text('''
events {}
http {
    include /test/site.conf;
    server {
        listen 8081;
        location / {
            proxy_pass http://127.0.0.1:80;
            proxy_set_header Host dashboard.test;
            proxy_set_header X-Forwarded-For $http_x_test_client;
        }
    }
    server {
        listen 8000;
        default_type application/json;
        if ($http_x_test_account_limit) {
            return 429 '{"error":{"message":"Account limit"}}';
        }
        return 200 '{"upstream":true}';
    }
    server { listen 3000; return 200; }
}
''')
    name = 'argus-login-edge-' + uuid4().hex[:12]
    def docker(*args):
        return subprocess.run(['docker', *args], check=True, text=True, capture_output=True).stdout.strip()
    docker('run', '--rm', '-d', '--name', name,
           '-p', '127.0.0.1::80', '-p', '127.0.0.1::8081',
           '--add-host', 'api:127.0.0.1', '--add-host', 'frontend:127.0.0.1',
           '-v', f'{directory}:/test:ro',
           '-v', f'{directory / "default.conf"}:/etc/nginx/includes/default.conf:ro',
           '-v', f'{directory / "log"}:/var/www/log',
           image, 'nginx', '-c', '/test/nginx.conf', '-g', 'daemon off;')
    try:
        ports = {port: docker('port', name, f'{port}/tcp').rsplit(':', 1)[1] for port in (80, 8081)}
        opener = build_opener(ProxyHandler({}))
        def request(path='/api/v1/dashboard/login', *, client='198.51.100.1', method='POST', direct=False, headers=None):
            port = ports[80 if direct else 8081]
            req = Request(f'http://127.0.0.1:{port}{path}', method=method,
                          headers={'Host': 'dashboard.test', 'X-Test-Client': client, **(headers or {})})
            try:
                response = opener.open(req, timeout=3)
            except HTTPError as exc:
                response = exc
            with response:
                return response.status, response.headers, response.read()
        for _ in range(50):
            try:
                if request('/api/ping', method='GET')[0] == 200:
                    break
            except (URLError, ConnectionError):
                pass
            time.sleep(.1)
        else:
            pytest.fail(docker('logs', name))
        yield request
    finally:
        docker('stop', name)


def test_login_ip_limit_is_independent_and_returns_json(edge):
    for _ in range(10):
        assert edge()[0] == 200
    status, headers, body = edge()
    assert status == 429
    assert headers['Retry-After'] == '10'
    assert headers['Cache-Control'] == 'no-store'
    assert json.loads(body)['error']['code'] == 'RATE_LIMITED'
    assert edge(client='198.51.100.2')[0] == 200
    # Account-limit responses from the API must retain their own explanation.
    status, _, body = edge(client='198.51.100.2', headers={'X-Test-Account-Limit': '1'})
    assert status == 429
    assert json.loads(body)['error']['message'] == 'Account limit'


def test_normalized_login_paths_share_limit_but_other_requests_do_not(edge):
    paths = ['/api/v1/dashboard/login?attempt=1', '/api/v1/dashboard/login/', '/api/v1/dashboard/%6cogin']
    for i in range(10):
        assert edge(paths[i % len(paths)], client='198.51.100.3')[0] == 200
    for path in paths:
        assert edge(path, client='198.51.100.3')[0] == 429
    for _ in range(15):
        assert edge('/api/v1/dashboard/me', client='198.51.100.3', method='GET')[0] == 200
        assert edge('/api/v1/dashboard/logout', client='198.51.100.3')[0] == 200
        assert edge(client='198.51.100.3', method='GET')[0] == 200


def test_untrusted_clients_cannot_rotate_forwarded_headers_to_bypass_limit(edge):
    for i in range(11):
        status, _, _ = edge(direct=True, headers={'X-Forwarded-For': f'203.0.113.{i+1}',
                                                  'X-Real-IP': f'203.0.113.{i+1}'})
        assert status == (200 if i < 10 else 429)


def test_trusted_proxy_uses_actual_client_after_attacker_supplied_chain(edge):
    for i in range(11):
        # The edge appends the actual source after any caller-provided values.
        status, _, _ = edge(client=f'203.0.113.{i+1}, 198.51.100.4')
        assert status == (200 if i < 10 else 429)
