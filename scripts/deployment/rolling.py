#!/usr/bin/env python3
"""Sequential HTTP replacement with durable per-service routing checkpoints."""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from release import artifact_digest

HTTP = {'clio': 8000, 'prophet-api': 8000, 'intelligence-api': 8000, 'api': 8000, 'frontend': 3000}
WORKERS = ['clio-worker', 'prophet', 'intelligence', 'proswin']


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.next')
    with open(temporary, 'w', opener=lambda name, flags: os.open(name, flags, 0o600)) as output:
        json.dump(value, output, indent=2)
        os.fchmod(output.fileno(), 0o600)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def slot_name(service, entry):
    return service + '-' + entry['slot'] if entry['slot'] else service


def router(service, spec, root):
    return {'image': 'nginx:stable-alpine', 'restart': 'unless-stopped',
            'networks': copy.deepcopy(spec['networks']),
            'volumes': [{'type': 'bind', 'source': str(root / 'routes' / service),
                         'target': '/etc/nginx/conf.d', 'read_only': True}],
            'stop_grace_period': '60s'}


def runtime_config(desired, state, root, pending=None):
    config = copy.deepcopy(desired)
    for service, entry in state['active'].items():
        config['services'].pop(service, None)
        if entry['slot']:
            config['services'][service] = router(service, entry['spec'], root)
        for item in (entry, entry.get('previous')):
            if item:
                config['services'][slot_name(service, item)] = item['spec']
    if pending:
        service, entry = pending['service'], pending['candidate']
        config['services'][slot_name(service, entry)] = entry['spec']
    # Routers are managed separately and must never be recreated by worker updates.
    for spec in config['services'].values():
        spec.pop('depends_on', None)
    return config


def route_text(service, target):
    port = HTTP[service]
    return f'''resolver 127.0.0.11 valid=5s ipv6=off;
server {{
    listen {port};
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


class Deployment:
    def __init__(self, root, bundle):
        self.root, self.bundle = root, bundle
        self.state_path = root / '.rolling-state.json'
        self.runtime = root / '.rolling-compose.json'
        self.base = ['docker', 'compose', '--project-directory', str(root),
                     '--env-file', str(root / '.env'), '--env-file', str(root / '.env.local')]
        self.desired = json.loads(self.capture(*self.base, '--env-file', str(bundle / 'images.env'),
                                               '-f', str(bundle / 'docker-compose.yml'), 'config', '--format', 'json'))
        environment = self.desired['services']['api']['environment']
        self.credentials = {key: environment.pop(key, '').replace('$$', '$') for key in ('DEPLOY_SMOKE_USERNAME', 'DEPLOY_SMOKE_PASSWORD')}
        self.minimum_available = int(environment.pop('DEPLOY_MIN_AVAILABLE_MIB', '1024'))
        if not all(self.credentials.values()):
            raise RuntimeError('Set DEPLOY_SMOKE_USERNAME and DEPLOY_SMOKE_PASSWORD in .env.local before deployment')
        self.manifest = json.loads((bundle / 'release.json').read_text())
        self.fingerprints = dict(line.split() for line in (bundle / 'fingerprints.tsv').read_text().splitlines())
        previous = root / '.release-fingerprints.tsv'
        self.previous_fingerprints = dict(line.split() for line in previous.read_text().splitlines()) if previous.exists() else {}
        self.migrations = [domain for domain in ('api', 'clio', 'prophet', 'intelligence')
                           if self.fingerprints[domain] != self.previous_fingerprints.get(domain)]
        approvals = self.manifest.get('online_migrations') or {}
        if self.state_path.exists():
            for domain in self.migrations:
                if approvals.get(domain) != self.fingerprints[domain]:
                    raise RuntimeError(f'{domain}: migration fingerprint must be approved in .deploy/online-migrations.json')
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else None

    @staticmethod
    def capture(*args):
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout

    @staticmethod
    def run(*args, **kwargs):
        subprocess.run(args, check=True, **kwargs)

    def compose(self, *args, capture=False, **kwargs):
        command = self.base + ['-f', str(self.runtime)] + list(args)
        if capture:
            return self.capture(*command)
        self.run(*command, **kwargs)

    def save(self):
        write_json(self.state_path, self.state)

    def install_runtime(self, pending=None):
        write_json(self.runtime, runtime_config(self.desired, self.state, self.root, pending))
        self.compose('config', '--quiet')

    def route(self, service, target):
        directory = self.root / 'routes' / service
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / 'default.conf.next'
        with temporary.open('w') as output:
            output.write(route_text(service, target))
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(directory / 'default.conf')

    def reload(self, service):
        workers = self.worker_pids(service)
        self.compose('exec', '-T', service, 'nginx', '-t')
        self.compose('exec', '-T', service, 'nginx', '-s', 'reload')
        return workers

    def worker_pids(self, service):
        output = self.compose('exec', '-T', service, 'ps', '-o', 'pid,args', capture=True)
        return {line.split()[0] for line in output.splitlines() if 'nginx: worker process' in line}

    def smoke(self, service, slot, routed=False):
        if service == 'frontend':
            # Node is available in the standalone frontend; check HTML and a referenced asset.
            target = service if routed else '127.0.0.1'
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
            self.compose('exec', '-T', slot, 'node', '--input-type=module', '-e', program,
                         f'http://{target}:3000', slot if routed else '')
            return
        source = (self.bundle / 'smoke.py').read_text()
        options = {'service': service, 'base': f'http://{service if routed else "127.0.0.1"}:8000',
                   'username': self.credentials['DEPLOY_SMOKE_USERNAME'],
                   'password': self.credentials['DEPLOY_SMOKE_PASSWORD'], 'expected_upstream': slot if routed else None}
        # Credentials travel over stdin, never process arguments or logs.
        self.compose('exec', '-T', slot, '.venv/bin/python', '-', input=source + '\ncheck(' + repr(options) + ')\n', text=True)

    def recover(self):
        pending = self.state.get('pending')
        if not pending:
            return
        service, candidate = pending['service'], pending['candidate']
        previous = self.state['active'][service]
        print(f'[deploy] Restoring {service} to its last checkpoint', flush=True)
        self.install_runtime(pending)
        if previous['slot']:
            old = slot_name(service, previous)
            self.compose('up', '-d', '--no-deps', '--wait', '--wait-timeout', '300', old)
            self.route(service, old)
            workers = self.reload(service)
            self.drain_router(service, workers)
        else:
            # Initial transition may already have replaced the legacy container with a router.
            self.compose('up', '-d', '--no-deps', '--wait', '--wait-timeout', '300', service)
        self.compose('stop', slot_name(service, candidate))
        self.state.pop('pending')
        self.save()
        self.install_runtime()

        self.smoke(service, slot_name(service, previous), routed=bool(previous['slot']))

    def snapshot(self):
        # Create writable bind sources before Docker can create them as root.
        (self.root / 'data/next-static').mkdir(parents=True, exist_ok=True)
        destination = self.root / 'releases' / self.fingerprints['configs']
        destination.mkdir(parents=True, exist_ok=True)
        if not (destination / 'configs').exists():
            temporary = destination / 'configs.next'
            shutil.copytree(self.bundle / 'configs', temporary, dirs_exist_ok=True)
            temporary.replace(destination / 'configs')
        artifacts = self.manifest.get('artifacts')
        if artifacts and (not isinstance(artifacts.get('id'), str) or len(artifacts['id']) != 64
                          or any(c not in '0123456789abcdef' for c in artifacts['id'])):
            raise RuntimeError('Invalid artifact identity')
        artifact_root = self.root / 'artifacts' / artifacts['id'] if artifacts else self.root / 'data'
        if artifacts and artifact_digest(artifact_root) != artifacts['id']:
            raise RuntimeError('Release artifacts are missing or corrupt')
        if not artifacts:
            # Freeze pre-transition models/metrics; later releases upload content-addressed bundles.
            artifact_root = self.root / 'releases' / 'legacy-artifacts'
            for directory in ('models', 'metrics'):
                target = artifact_root / directory
                if not target.exists():
                    shutil.copytree(self.root / 'data' / directory, target)
        for service in self.desired['services'].values():
            for volume in service.get('volumes', []):
                if volume.get('type') != 'bind':
                    continue
                source = Path(volume['source'])
                if source == self.root / 'configs':
                    volume['source'] = str(destination / 'configs')
                for directory in ('models', 'metrics'):
                    if source == self.root / 'data' / directory:
                        target = artifact_root / directory
                        if not target.is_dir():
                            raise RuntimeError(f'Missing release artifacts: {target}')
                        volume['source'] = str(target)
        # CLI one-off commands use the same versioned configuration as workers.
        write_json(self.root / '.release-compose.json', self.desired)
        demo = self.root / 'data/demo/current.json'
        if artifacts and not demo.exists():
            demo.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact_root / 'demo/current.json', demo)

    def backup_and_migrate(self):
        if not self.migrations:
            return
        backup_dir = self.root / 'backups'
        backup_dir.mkdir(exist_ok=True)
        path = backup_dir / time.strftime('pre-migration-%Y%m%dT%H%M%S.sql.gz', time.gmtime())
        partial = path.with_name(path.name + '.partial')
        with open(partial, 'wb', opener=lambda name, flags: os.open(name, flags, 0o600)) as output:
            process = subprocess.Popen(self.base + ['-f', str(self.runtime), 'exec', '-T', 'postgres',
                'sh', '-c', 'pg_dumpall --username="$POSTGRES_USER"'], stdout=subprocess.PIPE)
            with gzip.GzipFile(fileobj=output, mode='wb') as compressed:
                shutil.copyfileobj(process.stdout, compressed)
            if process.wait():
                raise RuntimeError('Database backup failed')
        partial.replace(path)
        # Readers and request writers remain online only for explicitly approved migrations.
        workers = {'api': [], 'clio': ['clio-worker'], 'prophet': ['prophet'], 'intelligence': ['intelligence']}
        for domain in self.migrations:
            if workers[domain]:
                self.state.setdefault('drained_workers', []).extend(workers[domain])
                self.save()
                self.compose('stop', *workers[domain])
            try:
                self.compose('run', '--rm', '--no-deps', domain + '-migrate')
            except Exception:
                self.resume_workers()
                raise

    def resume_workers(self):
        if self.state and self.state.get('drained_workers'):
            self.compose('start', *self.state['drained_workers'])
            self.state.pop('drained_workers')
            self.save()

    def check_memory(self, service):
        available = next(int(line.split()[1]) // 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        if available < self.minimum_available:
            raise RuntimeError(f'{service}: only {available} MiB available; need {self.minimum_available} MiB before starting a replacement')

    def replace(self, service):
        previous = self.state['active'][service]
        spec = copy.deepcopy(self.desired['services'][service])
        spec.pop('container_name', None)
        spec.pop('depends_on', None)
        # Avoid giving application slots the stable router's DNS aliases.
        for network in spec.get('networks', {}).values():
            if network:
                network.pop('aliases', None)
        if previous['slot'] and previous['spec'] == spec:
            return
        self.check_memory(service)
        candidate = {'slot': 'green' if previous['slot'] == 'blue' else 'blue', 'spec': spec}
        pending = {'service': service, 'candidate': candidate}
        self.state['pending'] = pending
        self.save()
        self.install_runtime(pending)
        name = slot_name(service, candidate)
        print(f'[deploy] Preparing {name}', flush=True)
        try:
            self.compose('pull', '--policy', 'missing', name)
            self.check_memory(service)
            self.compose('up', '-d', '--no-deps', '--wait', '--wait-timeout', '300', name)
            self.smoke(service, name)
            if service == 'frontend':
                self.publish_static(name)
            self.route(service, name)
            if previous['slot']:
                workers = self.reload(service)
            else:
                config = runtime_config(self.desired, self.state, self.root, pending)
                config['services'][service] = router(service, spec, self.root)
                write_json(self.runtime, config)
                self.compose('up', '-d', '--no-deps', service)
            self.smoke(service, name, routed=True)
            # QUIT makes old nginx workers finish existing requests. Do not stop the
            # old app until those workers have exited; a timeout rolls routing back.
            if previous['slot']:
                self.drain_router(service, workers)
                self.compose('stop', slot_name(service, previous))
            self.state['active'][service] = {**candidate, 'previous': {k: previous[k] for k in ('slot', 'spec')} if previous['slot'] else None}
            self.state.pop('pending')
            self.save()
            self.install_runtime()
        except BaseException:
            self.recover()
            raise

    def drain_router(self, service, workers):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if not workers & self.worker_pids(service):
                return
            time.sleep(1)
        raise RuntimeError(f'{service}: old router requests did not drain in 120 seconds')

    def publish_static(self, slot):
        target = self.root / 'data' / 'next-static'
        target.mkdir(parents=True, exist_ok=True)
        container = self.compose('ps', '-q', slot, capture=True).strip()
        staging = self.root / 'releases' / identity(self.manifest) / 'next-static'
        staging.mkdir(parents=True, exist_ok=True)
        self.run('docker', 'cp', container + ':/var/www/apps/web/.next/static/.', str(staging))
        self.run('rsync', '-a', '--ignore-existing', str(staging) + '/', str(target) + '/')

    def apply(self):
        self.snapshot()
        if self.state is None:
            installed = json.loads(self.capture(*self.base, '--env-file', str(self.root / '.release-images.env'),
                '-f', str(self.root / 'docker-compose.yml'), 'config', '--format', 'json'))
            self.state = {'active': {service: {'slot': None, 'spec': installed['services'][service]} for service in HTTP}}
            self.save()
        self.recover()
        self.resume_workers()
        self.install_runtime()
        # Download before starting a second HTTP instance; retain stopped rollback slots.
        self.compose('pull', '--policy', 'missing', *[s for s in self.desired['services'] if s not in HTTP])
        self.compose('up', '-d', '--no-deps', '--wait', 'postgres', 'redis')
        self.backup_and_migrate()
        for service in HTTP:
            self.replace(service)
        self.compose('up', '-d', '--no-deps', '--wait', '--wait-timeout', '300', *WORKERS)
        self.state.pop('drained_workers', None)
        self.save()
        # The generated HTTP template must not overwrite the live proxy configuration.
        for directory in ('configs', 'nginx', 'alloy'):
            self.run('rsync', '-a', '--delete', '--inplace', str(self.bundle / directory) + '/', str(self.root / directory) + '/')
        if self.fingerprints['nginx'] != self.previous_fingerprints.get('nginx'):
            # envsubst is normally startup-only; render updated templates before graceful reload.
            self.compose('exec', '-T', 'nginx', 'sh', '/docker-entrypoint.d/20-envsubst-on-templates.sh')
        self.compose('up', '-d', '--no-deps', 'nginx', 'alloy')
        self.reload('nginx')
        if self.fingerprints['alloy'] != self.previous_fingerprints.get('alloy'):
            self.compose('restart', 'alloy')
        for name in ('run', 'logs'):
            shutil.copy2(self.bundle / 'scripts' / name, self.root / 'scripts' / name)
        shutil.copy2(self.bundle / 'argus', self.root / 'argus')
        shutil.copy2(self.bundle / 'docker-compose.yml', self.root / 'docker-compose.yml')
        shutil.copyfile(self.bundle / 'images.env', self.root / '.release-images.env')
        for name in ('fingerprints.tsv', 'release.json'):
            destination = self.root / ('.release-fingerprints.tsv' if name == 'fingerprints.tsv' else '.release.json')
            temporary = destination.with_name(destination.name + '.next')
            shutil.copyfile(self.bundle / name, temporary)
            temporary.replace(destination)
        backups = sorted((self.root / 'backups').glob('pre-migration-*.sql.gz'), reverse=True)
        if self.migrations:
            for backup in backups[7:]:
                backup.unlink()
        print('[deploy] Release applied successfully', flush=True)
        # Stopped previous slots protect their images from pruning.
        try:
            self.run('docker', 'image', 'prune', '-a', '-f', '--filter', 'until=168h')
        except subprocess.CalledProcessError:
            print('[deploy] Warning: image cleanup failed', file=sys.stderr)


if __name__ == '__main__':
    deployment = None
    try:
        if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and sys.argv[3] != '--preflight'):
            raise RuntimeError('Usage: rolling.py ROOT BUNDLE [--preflight]')
        deployment = Deployment(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
        if len(sys.argv) == 3:
            deployment.apply()
        elif deployment.state is None:
            deployment.snapshot()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        if deployment is not None:
            deployment.resume_workers()
        # Never print captured Compose output: normalized configuration contains credentials.
        print(f'[deploy] Failed: {error}', file=sys.stderr)
        sys.exit(1)
