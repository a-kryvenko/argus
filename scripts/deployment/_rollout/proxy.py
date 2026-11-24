"""Switch nginx routes and wait for requests on the old route to finish."""
import os
import re
import time

from .applications import HTTP, slot_names

ROUTER = 'internal-router'


class NginxProxy:
    def __init__(self, root, compose, applications):
        self.root = root
        self.compose = compose
        self.applications = applications
        self.keepalive = True

    def route(self, service, target, *, public_api=None):
        self.write_route(self.root / 'routes/internal' / (service + '.conf'),
                         route_text(service, target, keepalive=self.keepalive, public_api=public_api))

    def write_color(self, color, *, split_public):
        targets = dict(zip(HTTP, slot_names(color)))
        public_api = targets.pop('public-api')
        for service, name in targets.items():
            self.route(service, name, public_api=public_api if service == 'api' and split_public else None)

    @staticmethod
    def write_route(path, content):
        directory = path.parent
        directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.next')
        with temporary.open('w') as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)

    def prepare(self, color, *, split_public=True):
        # Compose can start a previously stopped router with its original image.
        self.keepalive = False
        self.write_color(color, split_public=split_public)

        self.compose.run('up', '-d', '--no-deps', '--no-recreate', '--pull', 'never',
                     '--wait', '--wait-timeout', '300', ROUTER)
        version = self.compose.run('exec', '-T', ROUTER, 'sh', '-c', 'nginx -v 2>&1', capture=True)
        match = re.search(r'nginx/(\d+)\.(\d+)\.(\d+)', version)
        self.keepalive = bool(match and tuple(map(int, match.groups())) >= (1, 27, 3))
        if self.keepalive:
            self.write_color(color, split_public=split_public)
        else:
            print('[deploy] Internal router needs Nginx 1.27.3+ for upstream keep-alive with Docker DNS', flush=True)
        self.drain(ROUTER, self.reload(ROUTER))

    def switch(self, color, *, split_public=True):
        self.write_color(color, split_public=split_public)
        
        workers = self.reload(ROUTER)
        for service, name in zip(HTTP, slot_names(color)):
            if service == 'public-api' and not split_public:
                continue
            self.applications.check(service, name, routed=True)
        
        self.drain(ROUTER, workers)

    def reload_public(self):
        config_root = self.root / '.docker/nginx'
        for source, destination in (
            ('default.conf', '/etc/nginx/includes/default.conf'),
            ('templates/http.conf.template', '/etc/nginx/templates/website.conf.template'),
        ):
            content = (config_root / source).read_text()
            mounted = self.compose.run('exec', '-T', 'nginx', 'cat', destination, capture=True)
            if mounted != content:
                # Write through the existing bind mount so older containers see the release.
                self.compose.run('exec', '-T', 'nginx', 'sh', '-c', 'cat > "$1"', 'nginx-config',
                                 destination, input=content, text=True)
        self.compose.run('exec', '-T', 'nginx', 'sh', '/docker-entrypoint.d/20-envsubst-on-templates.sh')
        self.drain('nginx', self.reload('nginx'))

    def reload(self, service):
        workers = self.worker_pids(service)
        self.compose.run('exec', '-T', service, 'nginx', '-t')
        self.compose.run('exec', '-T', service, 'nginx', '-s', 'reload')
        
        return workers

    def worker_pids(self, service):
        output = self.compose.run('exec', '-T', service, 'ps', '-o', 'pid,args', capture=True)
        return {line.split()[0] for line in output.splitlines() if 'nginx: worker process' in line}

    def drain(self, service, workers):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if not workers & self.worker_pids(service):
                return
            time.sleep(1)
        
        raise RuntimeError(f'{service}: old router requests did not drain in 120 seconds')


def upstream_config(target, port, keepalive, variable):
    upstream = 'argus_' + target.replace('-', '_')
    resolver = '    resolver 127.0.0.11 valid=5s ipv6=off;\n'
    if keepalive:
        upstream_block = f'''upstream {upstream} {{
    zone {upstream} 64k;
{resolver}    resolver_timeout 5s;
    server {target}:{port} resolve;
    keepalive 16;
}}

'''
        server_resolver = ''
        backend = f'''        proxy_pass http://{upstream};
        proxy_set_header Connection "";
'''
    else:
        upstream_block = ''
        server_resolver = resolver
        backend = f'''        set ${variable} {target}:{port};
        proxy_pass http://${variable};
'''
    return upstream_block, server_resolver, backend


def location(route, backend, target):
    return f'''    location {route} {{
        add_header X-Argus-Upstream {target} always;
{backend}        proxy_http_version 1.1;
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $http_x_real_ip;
        proxy_set_header X-Forwarded-For $http_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $http_x_forwarded_proto;
        proxy_connect_timeout 5s;
        proxy_read_timeout 120s;
        proxy_buffering off;
    }}
'''


def route_text(service, target, *, keepalive=True, public_api=None):
    settings = HTTP[service]
    upstreams, resolver, backend = upstream_config(target, settings.port, keepalive, 'backend')
    locations = location('/', backend, target)
    if public_api:
        extra_upstream, _, public_backend = upstream_config(public_api, HTTP['public-api'].port, keepalive, 'public_backend')
        upstreams += extra_upstream
        locations += location('~ ^/api/v1/(?!dashboard(?:/|$))', public_backend, public_api)
    return f'''{upstreams}server {{
{resolver}    listen {settings.router_host}:{settings.router_port} bind;
{locations.rstrip()}
}}
'''

