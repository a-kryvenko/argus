"""Read merged Compose configurations without credentials or a Docker daemon."""
from functools import cache
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@cache
def load_compose(override):
    if not shutil.which('docker'):
        pytest.skip('Docker Compose is required for configuration checks')
    command = ['docker', 'compose', '--project-directory', str(ROOT), '--env-file', '/dev/null',
               '--profile', '*', '-f', str(ROOT / 'docker-compose.yml'), '-f', str(ROOT / override),
               'config', '--no-interpolate', '--format', 'json']
    config = json.loads(subprocess.check_output(command, text=True))
    for service in config['services'].values():
        environment = service.get('environment')
        if isinstance(environment, list):
            service['environment'] = dict(item.split('=', 1) for item in environment)
        if 'volumes' in service:
            mounts = []
            for mount in service['volumes']:
                source = mount['source']
                if mount['type'] == 'bind' and Path(source).is_relative_to(ROOT):
                    source = './' + str(Path(source).relative_to(ROOT))
                mounts.append(source + ':' + mount['target'] + (':ro' if mount.get('read_only') else ''))
            service['volumes'] = mounts
    return config
