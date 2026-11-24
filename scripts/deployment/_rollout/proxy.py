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

    def route(self, service, target):
        self.write_route(self.root / 'routes/internal' / (service + '.conf'),
                         route_text(service, target, keepalive=self.keepalive))

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

    def prepare(self, color):
        # Compose can start a previously stopped router with its original image.
        self.keepalive = False
        targets = list(zip(HTTP, slot_names(color)))
        for service, name in targets:
            self.route(service, name)

        self.compose.run('up', '-d', '--no-deps', '--no-recreate', '--pull', 'never',
                     '--wait', '--wait-timeout', '300', ROUTER)
        version = self.compose.run('exec', '-T', ROUTER, 'sh', '-c', 'nginx -v 2>&1', capture=True)
        match = re.search(r'nginx/(\d+)\.(\d+)\.(\d+)', version)
        self.keepalive = bool(match and tuple(map(int, match.groups())) >= (1, 27, 3))
        if self.keepalive:
            for service, name in targets:
                self.route(service, name)
        else:
            print('[deploy] Internal router needs Nginx 1.27.3+ for upstream keep-alive with Docker DNS', flush=True)
        self.drain(ROUTER, self.reload(ROUTER))

    def switch(self, color):
        for service, name in zip(HTTP, slot_names(color)):
            self.route(service, name)
        
        workers = self.reload(ROUTER)
        for service, name in zip(HTTP, slot_names(color)):
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


def route_text(service, target, *, keepalive=True):
    settings = HTTP[service]
    upstream = 'argus_' + target.replace('-', '_')
    listen = f'{settings.router_host}:{settings.router_port} bind'
    resolver = '    resolver 127.0.0.11 valid=5s ipv6=off;\n'
    if keepalive:
        upstream_block = f'''upstream {upstream} {{
    zone {upstream} 64k;
{resolver}    resolver_timeout 5s;
    server {target}:{settings.port} resolve;
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
        backend = f'''        set $backend {target}:{settings.port};
        proxy_pass http://$backend;
'''
    return f'''{upstream_block}server {{
{server_resolver}    listen {listen};
    add_header X-Argus-Upstream {target} always;
    location / {{
{backend}        proxy_http_version 1.1;
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $http_x_real_ip;
        proxy_set_header X-Forwarded-For $http_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $http_x_forwarded_proto;
        proxy_connect_timeout 5s;
        proxy_read_timeout 120s;
        proxy_buffering off;
    }}
}}
'''

