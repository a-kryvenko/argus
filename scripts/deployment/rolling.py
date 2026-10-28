#!/usr/bin/env python3
"""Sequential HTTP replacement with durable per-service routing checkpoints."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
from release import artifact_digest
from backup import create_backup

HTTP = {'clio': 8000, 'prophet-api': 8000, 'intelligence-api': 8000, 'api': 8000, 'frontend': 3000}
WORKERS = ['intelligence', 'proswin']
ROUTER = 'internal-router'
ROUTES = {
    'clio': ('observations', 'observations-router', 8101),
    'prophet-api': ('forecasts', 'forecasts-router', 8102),
    'intelligence-api': ('impacts', 'impacts-router', 8103),
    'api': ('application', 'application-router', 8104),
    'frontend': ('application', 'application-router', 8105),
}


def routed_url(service):
    _, host, port = ROUTES[service]
    return f'http://{host}:{port}'


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
    return service + '-' + entry['slot']


def legacy_router(service, spec, root):
    return {'image': 'nginx:stable-alpine', 'restart': 'unless-stopped',
            'networks': copy.deepcopy(spec['networks']),
            'volumes': [{'type': 'bind', 'source': str(root / 'routes' / service),
                         'target': '/etc/nginx/conf.d', 'read_only': True}],
            'stop_grace_period': '60s'}


def router(state, root):
    networks = {ROUTES[service][0]: {'aliases': [ROUTES[service][1]]}
                for service in state['active']}
    return {'image': 'nginx:stable-alpine', 'restart': 'unless-stopped',
            'networks': networks,
            'volumes': [{'type': 'bind', 'source': str(root / 'routes/internal'),
                         'target': '/etc/nginx/conf.d', 'read_only': True}],
            'healthcheck': {'test': ['CMD', 'nginx', '-t'], 'interval': '10s',
                            'timeout': '5s', 'retries': 3, 'start_period': '10s', 'start_interval': '1s'},
            'stop_grace_period': '120s'}


def runtime_config(desired, state, root, pending=None):
    config = copy.deepcopy(desired)
    for service, entry in state['active'].items():
        config['services'].pop(service, None)
        if state.get('router_version', 1) < 2:
            config['services'][service] = legacy_router(service, entry['spec'], root)
        for item in (entry, entry.get('previous')):
            if item:
                config['services'][slot_name(service, item)] = item['spec']
    if pending:
        service, entry = pending['service'], pending['candidate']
        config['services'][slot_name(service, entry)] = entry['spec']
    config['services'][ROUTER] = router(state, root)
    # Routers are managed separately and must never be recreated by worker updates.
    for spec in config['services'].values():
        spec.pop('depends_on', None)
    return config


def route_text(service, target, *, legacy=False):
    port = HTTP[service]
    _, host, listener = ROUTES[service]
    listen = str(port) if legacy else f'{host}:{listener} bind'
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


class Deployment:
    def __init__(self, root, bundle):
        self.root, self.bundle = root, bundle
        self.state_path = root / '.rolling-state.json'
        if not self.state_path.is_file():
            raise RuntimeError('Missing .rolling-state.json; restore deployment state before applying a release')
        try:
            self.state = json.loads(self.state_path.read_text())
        except ValueError:
            raise RuntimeError('Invalid rolling deployment state') from None
        if (not isinstance(self.state, dict) or not isinstance(self.state.get('active'), dict)
                or set(self.state['active']) != set(HTTP)
                or any(not isinstance(entry, dict) or entry.get('slot') not in ('blue', 'green')
                       or not isinstance(entry.get('spec'), dict) for entry in self.state['active'].values())):
            raise RuntimeError('Incomplete rolling deployment state; restore state before applying a release')
        self.runtime = root / '.rolling-compose.json'
        self.base = ['docker', 'compose', '--project-directory', str(root),
                     '--env-file', str(root / '.env'), '--env-file', str(root / '.env.local')]
        self.desired = json.loads(self.capture(*self.base, '--env-file', str(bundle / 'images.env'),
                                               '--profile', '*', '-f', str(bundle / 'docker-compose.yml'), 'config', '--format', 'json'))
        required = {domain + '-migrate' for domain in ('api', 'clio', 'prophet', 'intelligence')}
        if not required <= self.desired['services'].keys():
            raise RuntimeError('Release configuration is missing maintenance services')
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

    @staticmethod
    def capture(*args, **kwargs):
        return subprocess.run(args, check=True, capture_output=True, text=True, **kwargs).stdout

    @staticmethod
    def run(*args, **kwargs):
        subprocess.run(args, check=True, **kwargs)

    def compose(self, *args, capture=False, **kwargs):
        command = self.base + ['-f', str(self.runtime)] + list(args)
        if capture:
            return self.capture(*command, **kwargs)
        self.run(*command, **kwargs)

    def save(self):
        write_json(self.state_path, self.state)

    def install_runtime(self, pending=None):
        write_json(self.runtime, runtime_config(self.desired, self.state, self.root, pending))
        self.compose('config', '--quiet')

    def route(self, service, target):
        self.write_route(self.root / 'routes/internal' / (service + '.conf'), route_text(service, target))
        if self.state.get('router_version', 1) < 2:
            self.write_route(self.root / 'routes' / service / 'default.conf', route_text(service, target, legacy=True))

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

    def prepare_router(self):
        for service, entry in self.state['active'].items():
            self.route(service, slot_name(service, entry))
        self.install_runtime(self.state.get('pending'))
        self.compose('up', '-d', '--no-deps', '--wait', '--wait-timeout', '300', ROUTER)

    def reload_routes(self, service):
        workers = {ROUTER: self.reload(ROUTER)}
        if self.state.get('router_version', 1) < 2:
            workers[service] = self.reload(service)
        return workers

    def drain_routes(self, workers):
        for router_name, pids in workers.items():
            self.drain_router(router_name, pids)

    def retire_legacy_routers(self):
        # All readers and the public nginx now use the shared listeners. Persist
        # before removal so a retry never tries to reload an already removed proxy.
        self.state['router_version'] = 2
        self.save()
        self.install_runtime()
        for service in self.state['active']:
            self.retire_worker(service)

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
            self.compose('exec', '-T', slot, 'node', '--input-type=module', '-e', program,
                         base, slot if routed else '')
            return
        source = (self.bundle / 'smoke.py').read_text()
        options = {'service': service, 'base': routed_url(service) if routed else 'http://127.0.0.1:8000',
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
        old = slot_name(service, previous)
        self.compose('up', '-d', '--no-deps', '--wait', '--wait-timeout', '300', old)
        self.route(service, old)
        workers = self.reload_routes(service)
        self.drain_routes(workers)
        self.compose('stop', slot_name(service, candidate))
        self.state.pop('pending')
        self.save()
        self.install_runtime()

        self.smoke(service, slot_name(service, previous), routed=True)

    def snapshot(self):
        artifacts = self.manifest.get('artifacts')
        if (not isinstance(artifacts, dict) or not isinstance(artifacts.get('id'), str)
                or len(artifacts['id']) != 64 or any(c not in '0123456789abcdef' for c in artifacts['id'])):
            raise RuntimeError('Release must contain a valid immutable artifact identity')
        artifact_root = self.root / 'artifacts' / artifacts['id']
        if artifact_digest(artifact_root) != artifacts['id']:
            raise RuntimeError('Release artifacts are missing or corrupt')
        # Create writable bind sources before Docker can create them as root.
        (self.root / 'data/next-static').mkdir(parents=True, exist_ok=True)
        destination = self.root / 'releases' / self.fingerprints['configs']
        destination.mkdir(parents=True, exist_ok=True)
        if not (destination / 'configs').exists():
            temporary = destination / 'configs.next'
            shutil.copytree(self.bundle / 'configs', temporary, dirs_exist_ok=True)
            temporary.replace(destination / 'configs')
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
        demo = self.root / 'data/demo/current.json'
        demo.parent.mkdir(parents=True, exist_ok=True)
        output = tempfile.NamedTemporaryFile(dir=demo.parent, delete=False)
        temporary = Path(output.name)
        try:
            with output:
                with (artifact_root / 'demo/current.json').open('rb') as source:
                    shutil.copyfileobj(source, output)
                os.fchmod(output.fileno(), 0o644)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(demo)
        finally:
            temporary.unlink(missing_ok=True)
        directory = os.open(demo.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def backup_and_migrate(self):
        if not self.migrations:
            return
        create_backup(self.base + ['-f', str(self.runtime), 'exec', '-T', 'postgres',
            'sh', '-c', 'pg_dumpall --username="$POSTGRES_USER"'], self.root / 'backups')
        # Readers and request writers remain online only for compatible migrations.
        workers = {'api': [], 'clio': [], 'prophet': [], 'intelligence': ['intelligence']}
        for domain in self.migrations:
            if domain == 'prophet' and self.worker_containers('prophet'):
                workers[domain] = ['prophet']
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
            workers = [name for name in self.state['drained_workers'] if name in WORKERS or name == 'prophet']
            if workers:
                self.compose('start', *workers)
            self.state.pop('drained_workers')
            self.save()

    def check_pending_migrations(self):
        if not self.migrations:
            return
        source = (self.bundle / 'migration_status.py').read_text()
        pending = []
        for domain in self.migrations:
            status = self.compose('run', '--rm', '--no-deps', '-T', domain + '-migrate',
                                  '.venv/bin/python', '-', domain, '--online', capture=True, input=source).strip()
            if status.startswith('blocked:'):
                raise RuntimeError(f'{domain}: migrations require downtime or compatibility review: {status.removeprefix("blocked:")}; '
                                   'rolling deployment requires online_compatible = True in each pending migration')
            if status not in ('current', 'pending'):
                raise RuntimeError(f'{domain}: invalid migration status')
            if status == 'pending':
                pending.append(domain)
        self.migrations = pending
        print(f'[deploy] Pending database upgrades: {", ".join(pending) or "none"}', flush=True)

    def cleanup_backups(self):
        if not self.migrations:
            return
        backups = sorted({path for pattern in ('pre-migration-*.sql', 'pre-migration-*.sql.gz')
                          for path in (self.root / 'backups').glob(pattern) if path.is_file()}, reverse=True)
        for backup in backups[3:]:
            backup.unlink()

    def check_memory(self, service):
        available = next(int(line.split()[1]) // 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        if available < self.minimum_available:
            raise RuntimeError(f'{service}: only {available} MiB available; need {self.minimum_available} MiB before starting a replacement')

    def diagnose_candidate(self, name):
        print(f'[deploy] Failed candidate {name}: container state and recent logs', flush=True)
        try:
            containers = self.compose('ps', '--all', '-q', name, capture=True).split()
            for container in containers:
                # Inspect only runtime state; container configuration includes secrets.
                self.run('docker', 'inspect', '--format', '{{json .State}}', container)
                self.run('docker', 'logs', '--tail', '100', container)
        except Exception:
            print(f'[deploy] Could not collect diagnostics for {name}', file=sys.stderr)

    def replace(self, service):
        previous = self.state['active'][service]
        spec = copy.deepcopy(self.desired['services'][service])
        spec.pop('container_name', None)
        spec.pop('depends_on', None)
        # Avoid giving application slots the stable router's DNS aliases.
        for network in spec.get('networks', {}).values():
            if network:
                network.pop('aliases', None)
        if previous['spec'] == spec:
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
            workers = self.reload_routes(service)
            self.smoke(service, name, routed=True)
            # QUIT makes old nginx workers finish existing requests. Do not stop the
            # old app until those workers have exited; a timeout rolls routing back.
            self.drain_routes(workers)
            self.compose('stop', slot_name(service, previous))
            self.state['active'][service] = {**candidate, 'previous': {k: previous[k] for k in ('slot', 'spec')}}
            self.state.pop('pending')
            self.save()
            self.install_runtime()
        except BaseException:
            self.diagnose_candidate(name)
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

    def install_crontab(self):
        if any(char in str(self.root) for char in ('\n', '\r', '%')):
            raise RuntimeError('Cron deployment path must not contain newlines or %')
        paths = [self.bundle / 'apps' / service / 'cronjobs.txt'
                 for service in ('clio', 'prophet', 'api', 'intelligence')]
        paths.append(self.bundle / '.deploy/cronjobs.txt')
        contents = [path.read_text().rstrip() for path in paths]
        crontab = '\n\n'.join(contents).replace('@ROOT@', shlex.quote(str(self.root))) + '\n'
        self.run('crontab', '-', input=crontab, text=True)

    def retire_clio_worker(self):
        self.retire_worker('clio-worker')

    def worker_containers(self, service):
        return self.capture('docker', 'ps', '-aq', '--filter',
            f'label=com.docker.compose.project={self.desired["name"]}',
            '--filter', f'label=com.docker.compose.service={service}').split()

    def retire_worker(self, service):
        containers = self.worker_containers(service)
        if containers:
            self.run('docker', 'stop', '--time', '600', *containers)
            self.run('docker', 'rm', *containers)

    def apply(self):
        self.snapshot()
        self.prepare_router()
        self.recover()
        self.drain_router(ROUTER, self.reload(ROUTER))
        if self.state.get('router_version', 1) < 2:
            for service, entry in self.state['active'].items():
                self.smoke(service, slot_name(service, entry), routed=True)
        self.resume_workers()
        self.install_runtime()
        # Download before starting a second HTTP instance; retain stopped rollback slots.
        self.compose('pull', '--policy', 'missing', *[s for s in self.desired['services'] if s not in HTTP])
        self.check_pending_migrations()
        self.compose('up', '-d', '--no-deps', '--wait', 'postgres', 'redis')
        self.backup_and_migrate()
        for service in HTTP:
            self.replace(service)
        self.compose('up', '-d', '--no-deps', '--wait', '--wait-timeout', '300', *WORKERS)
        legacy = [name for name in self.state.get('drained_workers', []) if name == 'prophet']
        if legacy:
            self.state['drained_workers'] = legacy
        else:
            self.state.pop('drained_workers', None)
        self.save()
        # The generated HTTP template must not overwrite the live proxy configuration.
        for directory in ('nginx', 'alloy'):
            self.run('rsync', '-a', '--delete', '--inplace', str(self.bundle / directory) + '/', str(self.root / directory) + '/')
        if self.fingerprints['nginx'] != self.previous_fingerprints.get('nginx'):
            # envsubst is normally startup-only; render updated templates before graceful reload.
            self.compose('exec', '-T', 'nginx', 'sh', '/docker-entrypoint.d/20-envsubst-on-templates.sh')
        self.compose('up', '-d', '--no-deps', 'nginx', 'alloy')
        self.drain_router('nginx', self.reload('nginx'))
        if self.fingerprints['alloy'] != self.previous_fingerprints.get('alloy'):
            self.compose('restart', 'alloy')
        self.retire_legacy_routers()
        for name in ('run', 'logs', 'cron-run'):
            shutil.copy2(self.bundle / 'scripts' / name, self.root / 'scripts' / name)
        shutil.copy2(self.bundle / 'argus', self.root / 'argus')
        self.install_crontab()
        self.retire_clio_worker()
        self.retire_worker('prophet')
        self.state.pop('drained_workers', None)
        self.save()
        # Keep the previous CLI usable when preflight, migration or replacement fails.
        write_json(self.root / '.release-compose.json', self.desired)
        for name in ('fingerprints.tsv', 'release.json'):
            destination = self.root / ('.release-fingerprints.tsv' if name == 'fingerprints.tsv' else '.release.json')
            temporary = destination.with_name(destination.name + '.next')
            shutil.copyfile(self.bundle / name, temporary)
            temporary.replace(destination)
        self.cleanup_backups()
        print('[deploy] Release applied successfully', flush=True)
        # Stopped previous slots protect their images from pruning.
        try:
            self.run('docker', 'image', 'prune', '-a', '-f', '--filter', 'until=72h')
        except subprocess.CalledProcessError:
            print('[deploy] Warning: image cleanup failed', file=sys.stderr)


if __name__ == '__main__':
    deployment = None
    try:
        if len(sys.argv) != 3:
            raise RuntimeError('Usage: rolling.py ROOT BUNDLE')
        deployment = Deployment(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
        deployment.apply()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        if deployment is not None:
            deployment.resume_workers()
        # Never print captured Compose output: normalized configuration contains credentials.
        print(f'[deploy] Failed: {error}', file=sys.stderr)
        sys.exit(1)
