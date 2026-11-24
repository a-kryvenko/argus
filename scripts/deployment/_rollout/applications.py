"""Application slots, smoke checks and frontend static assets."""

from dataclasses import dataclass
from .compose import Compose


@dataclass(frozen=True)
class HttpService:
    port: int
    router_host: str
    router_port: int


HTTP = {
    'clio': HttpService(8000, 'observations-router', 8101),
    'prophet-api': HttpService(8000, 'forecasts-router', 8102),
    'intelligence-api': HttpService(8000, 'impacts-router', 8103),
    'api': HttpService(8000, 'application-router', 8104),
    'public-api': HttpService(8000, 'application-router', 8104),
    'frontend': HttpService(3000, 'application-router', 8105),
}
WORKERS = ('clio-worker', 'prophet-worker', 'intelligence', 'proswin')


def slot_names(color):
    return [f'{name}-{color}' for name in HTTP]


def routed_url(name):
    service = HTTP[name]
    return f'http://{service.router_host}:{service.router_port}'


class Applications:
    def __init__(self, root, compose: Compose):
        self.root = root
        self.compose = compose

    def check(self, service: str, slot, routed=False):
        if service == 'public-api':
            base = routed_url(service) if routed else 'http://127.0.0.1:8000'
            self.compose.run('exec', '-T', slot, 'public-api', 'smoke', base, slot if routed else '')
            return
        if service == 'frontend':
            # Node is available in the standalone frontend; check HTML and a referenced asset.
            base = routed_url(service) if routed else 'http://127.0.0.1:3000'
            program = '''const base=process.argv[1], expected=process.argv[2];
let response;
for(let i=0;i<50;i++) {
  response=await fetch(base);
  if(!expected || response.headers.get('x-argus-upstream')===expected) break;
  await new Promise(resolve=>setTimeout(resolve,100));
}
if(expected && response.headers.get('x-argus-upstream')!==expected) throw Error('Wrong frontend upstream');
if(!response.ok) throw Error('HTML status');
const html=await response.text(); if(!html.includes('<html')) throw Error('HTML body');
const asset=html.match(/(?:src|href)="([^" ]*\\/_next\\/static\\/[^" ]+)"/);
if(!asset) throw Error('Missing static asset');
const result=await fetch(new URL(asset[1].replaceAll('&amp;','&'),base));
if(!result.ok || !(await result.arrayBuffer()).byteLength) throw Error('Static asset');'''
            self.compose.run('exec', '-T', slot, 'node', '--input-type=module', '-e', program,
                         base, slot if routed else '')
            return
        source = (self.root / 'scripts/deployment/smoke.py').read_text()
        options = {'service': service, 'base': routed_url(service) if routed else 'http://127.0.0.1:8000',
                   'expected_upstream': slot if routed else None}
        command = ('.venv/bin/python', '-')
        if service in ('clio', 'prophet-api'):
            # A failed first Go rollout can route back to a running FastAPI slot.
            binary = {'clio': 'clio-api', 'prophet-api': 'prophet-api'}[service]
            command = ('sh', '-c',
                       f'if command -v {binary} >/dev/null; then exec {binary} smoke "$@"; fi; exec .venv/bin/python -',
                       f'{service}-smoke', options['base'], options['expected_upstream'] or '')
        self.compose.run('exec', '-T', slot, *command, input=source + '\ncheck(' + repr(options) + ')\n', text=True)

    def start(self, color):
        for service, name in zip(HTTP, slot_names(color)):
            self.compose.run('up', '-d', '--no-deps', '--force-recreate', '--pull', 'never',
                         '--wait', '--wait-timeout', '300', name)
            self.check(service, name)
            if service == 'frontend':
                self.publish_static(name)

    def publish_static(self, slot):
        target = self.root / 'data' / 'next-static'
        target.mkdir(parents=True, exist_ok=True)
        container = self.compose.run('ps', '-q', slot, capture=True).strip()
        staging = self.root / 'data/static-staging'
        staging.mkdir(parents=True, exist_ok=True)
        self.compose.execute('docker', 'cp', container + ':/var/www/apps/web/.next/static/.', str(staging))
        self.compose.execute('rsync', '-a', '--ignore-existing', str(staging) + '/', str(target) + '/')
