#!/usr/bin/env python3
"""Assemble a deployment bundle from local release files and data."""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from _rollout.checksum import get_checksum

ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORIES = ('models', 'metrics')
DEMO_FILE = Path('demo/current.json')

def _copy_release_files(destination):
    shutil.copytree(
        ROOT / '.deploy',
        destination,
        ignore=shutil.ignore_patterns('__pycache__', 'artifacts.json'),
    )
    
    (destination / 'docker-compose.yml').rename(destination / 'docker-compose.prod.yml')

    ignore = shutil.ignore_patterns('__pycache__')
    
    shutil.copytree(ROOT / 'configs', destination / 'configs', ignore=ignore)
    shutil.copytree(ROOT / 'scripts/deployment', destination / 'scripts/deployment', ignore=ignore)

    repository_files = (
        'docker-compose.yml',
        'argus',
        'scripts/run',
        'scripts/logs',
    )
    for name in repository_files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)


def _copy_data_artifacts(destination):
    data = ROOT / 'data'
    for name in DATA_DIRECTORIES:
        shutil.copytree(data / name, destination / name)
    demo = destination / DEMO_FILE
    demo.parent.mkdir()
    shutil.copy2(data / DEMO_FILE, demo)


def _assemble(destination, mode):
    _copy_release_files(destination)
    _copy_data_artifacts(destination / 'data')

    release_id = uuid4().hex

    return {
        'mode': mode,
        'release_id': release_id,
        'checksum': get_checksum(destination)
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maintenance', action='store_true', help='Build a maintenance release')
    args = parser.parse_args()

    output = args.output
    if output.exists():
        raise SystemExit(f'Output already exists: {output}')

    data = ROOT / 'data'
    for name in DATA_DIRECTORIES:
        if not (data / name).is_dir():
            raise SystemExit(f'Missing data/{name}')
    
    if not (data / DEMO_FILE).is_file():
        raise SystemExit(f'Missing data/{DEMO_FILE}')

    mode = 'maintenance' if args.maintenance else 'rolling'

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        bundle = Path(directory) / 'bundle'
        manifest = _assemble(bundle, mode)
        bundle.rename(output)

    (output.parent / '.release_manifest').write_text(json.dumps(manifest))
    (output.parent / '.release').write_text(manifest['release_id'] + '\n')
    print(manifest['release_id'])


if __name__ == '__main__':
    main()
