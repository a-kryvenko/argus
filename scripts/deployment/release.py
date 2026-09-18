#!/usr/bin/env python3
"""Build identities depend on actual image inputs, not the previous git tag."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess

IMAGES = {'frontend': 'argus-frontend', 'api': 'argus-forecast', 'clio': 'argus-clio', 'prophet': 'argus-prophet', 'intelligence': 'argus-intelligence'}
INPUTS = {
    'intelligence': ['apps/intelligence/pyproject.toml', 'apps/intelligence/uv.lock', 'apps/intelligence/src', 'packages/common'],
    'frontend': ['package.json', 'pnpm-lock.yaml', 'pnpm-workspace.yaml', 'apps/web'],
    'api': ['apps/api/pyproject.toml', 'apps/api/uv.lock', 'apps/api/app', 'apps/api/alembic',
            'apps/api/alembic.ini', 'packages/common', 'scripts/db/provision.py'],
    'clio': ['apps/clio/pyproject.toml', 'apps/clio/uv.lock', 'apps/clio/src', 'packages/common', 'packages/clio', 'packages/forecast'],
    'prophet': ['apps/prophet/pyproject.toml', 'apps/prophet/uv.lock', 'apps/prophet/src',
                'packages/common', 'packages/clio', 'packages/forecast'],
}
MIGRATIONS = {'intelligence': ['apps/intelligence/src/argus_intelligence/migrations'],
              'api': ['apps/api/alembic', 'apps/api/alembic.ini'],
              'clio': ['apps/clio/src/argus_clio/migrations'],
              'prophet': ['apps/prophet/src/argus_prophet/migrations']}
IGNORED = {'.git', '.venv', '__pycache__', '.pytest_cache', 'node_modules', '.next',
           '.turbo', 'dist', 'build', 'coverage', 'test-results', 'playwright-report'}


def digest(root, prefixes, *, private=False):
    paths = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z', '--', *prefixes]).decode().split('\0')
    result = hashlib.sha256()
    for name in sorted(set(paths) - {''}):
        path = Path(name)
        if any(part in IGNORED for part in path.parts) or (private and path.parts[0] in ('tests', 'examples')):
            continue
        if path.name == '.env' or path.name.startswith('.env.') or path.suffix in ('.log', '.tsbuildinfo'):
            continue
        full = root / name
        if not full.exists() and not full.is_symlink():
            continue
        content = os.readlink(full).encode() if full.is_symlink() else full.read_bytes()
        result.update(name.encode() + b'\0' + str(full.lstat().st_mode & 0o777).encode() + b'\0')
        result.update(hashlib.sha256(content).digest())
    return result.hexdigest()


def plan(root, *, rebuild=''):
    private = root / 'packages/forecast-core'
    private_sha = subprocess.check_output(['git', '-C', str(private), 'rev-parse', 'HEAD'], text=True).strip()
    private_hash = digest(private, ['.'], private=True)
    intelligence_core = root / 'packages/intelligence-core'
    intelligence_core_sha = subprocess.check_output(['git', '-C', str(intelligence_core), 'rev-parse', 'HEAD'], text=True).strip()
    intelligence_core_hash = digest(intelligence_core, ['.'], private=True)
    result = {'version': 1, 'commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
              'private_commit': private_sha, 'intelligence_core_commit': intelligence_core_sha, 'components': {}, 'migrations': {},
              'config_hash': digest(root, ['configs'])}
    for name, prefixes in INPUTS.items():
        source = digest(root, [*prefixes, '.dockerignore', f'.deploy/{name}.Dockerfile'])
        backend_hash = private_hash if name in ('clio', 'prophet') else intelligence_core_hash if name == 'intelligence' else ''
        identity = hashlib.sha256((source + backend_hash + rebuild).encode()).hexdigest()
        result['components'][name] = {'repository': 'ghcr.io/a-kryvenko/' + IMAGES[name], 'tag': 'src-' + identity}
    result['migrations'] = {name: digest(root, paths) for name, paths in MIGRATIONS.items()}
    return result


def bundle(root, plan_path, images_path, destination):
    manifest = json.loads(plan_path.read_text())
    manifest['images'] = {}
    for name, spec in manifest['components'].items():
        image = json.loads((images_path / f'{name}.json').read_text())
        if image['tag'] != spec['tag'] or not re.fullmatch(re.escape(spec['repository']) + r'@sha256:[0-9a-f]{64}', image['image']):
            raise ValueError(f'Image result does not match the plan: {name}')
        manifest['images'][name] = image['image']
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('docker-compose.yml', 'nginx-proxy-up.sh'):
        shutil.copy2(root / '.deploy' / name, destination / name)
    for name in ('nginx', 'alloy'):
        shutil.copytree(root / '.deploy' / name, destination / name, dirs_exist_ok=True)
    shutil.copytree(root / 'configs', destination / 'configs', dirs_exist_ok=True)
    shutil.copy2(root / 'scripts/deployment/deploy.sh', destination / 'deploy.sh')
    shutil.copy2(root / 'argus', destination / 'argus')
    shutil.copytree(root / 'scripts/prod', destination / 'scripts/prod', dirs_exist_ok=True)
    (destination / 'release.json').write_text(json.dumps(manifest, indent=2) + '\n')
    fingerprints = {**manifest['migrations'], 'configs': manifest['config_hash'],
                    'nginx': digest(root, ['.deploy/nginx']), 'alloy': digest(root, ['.deploy/alloy'])}
    (destination / 'fingerprints.tsv').write_text(''.join(f'{key} {value}\n' for key, value in fingerprints.items()))
    (destination / 'images.env').write_text(''.join(f'ARGUS_{name.upper()}_IMAGE={value}\n' for name, value in manifest['images'].items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('plan')
    prepare.add_argument('--rebuild', default='')
    prepare.add_argument('--output', type=Path, required=True)
    assemble = commands.add_parser('bundle')
    assemble.add_argument('--plan', type=Path, required=True)
    assemble.add_argument('--images', type=Path, required=True)
    assemble.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.command == 'plan':
        args.output.write_text(json.dumps(plan(root, rebuild=args.rebuild), indent=2) + '\n')
    else:
        bundle(root, args.plan, args.images, args.output)
