"""Switch nginx routes and wait for requests on the old route to finish."""
import os
import time

from .applications import HTTP, slot_names

ROUTER = 'internal-router'


class NginxProxy:
    def __init__(self, root, compose, applications):
        self.root = root
        self.compose = compose
        self.applications = applications

    def route(self, service, target):
        self.write_route(self.root / 'routes/internal' / (service + '.conf'), route_text(service, target))

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
        for service, name in zip(HTTP, slot_names(color)):
            self.route(service, name)

        self.compose.run('up', '-d', '--no-deps', '--no-recreate', '--pull', 'never',
                     '--wait', '--wait-timeout', '300', ROUTER)
        self.drain(ROUTER, self.reload(ROUTER))

    def switch(self, color):
        for service, name in zip(HTTP, slot_names(color)):
            self.route(service, name)
        
        workers = self.reload(ROUTER)
        for service, name in zip(HTTP, slot_names(color)):
            self.applications.check(service, name, routed=True)
        
        self.drain(ROUTER, workers)

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


def route_text(service, target):
    settings = HTTP[service]
    port = settings.port
    listen = f'{settings.router_host}:{settings.router_port} bind'
    return f'''server {{
    resolver 127.0.0.11 valid=5s ipv6=off;
    listen {listen};
    add_header X-Argus-Upstream {target} always;
    location / {{
        set $backend {target}:{port};
        proxy_pass http://$backend;
        proxy_http_version 1.1;
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

