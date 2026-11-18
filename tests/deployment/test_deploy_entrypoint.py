"""Exercise the real shell lock without SSH, Docker or production access."""
import fcntl
import os
from pathlib import Path
import select
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_rollout_entrypoint_loads_from_standalone_bundle(tmp_path):
    source = ROOT / 'scripts/deployment'
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    for name in ('rollout.py', 'backup.py'):
        shutil.copy2(source / name, bundle / name)
    shutil.copytree(source / '_rollout', bundle / '_rollout',
                    ignore=shutil.ignore_patterns('__pycache__'))
    root = tmp_path / 'uninitialized-host'

    result = subprocess.run(
        ['python3', str(bundle / 'rollout.py'), str(root), str(bundle)],
        cwd=tmp_path, env={**os.environ, 'PYTHONPATH': ''},
        capture_output=True, text=True, timeout=5,
    )

    assert result.returncode == 1
    assert '.release_manifest' in result.stderr
    assert not root.exists()


@pytest.fixture
def release(tmp_path, monkeypatch):
    root, tools = tmp_path / 'root', tmp_path / 'tools'
    bundle = root / 'incoming' / ('a' * 32)
    for path in (root, bundle, tools):
        path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / '.deploy/deploy.sh', root / 'deploy.sh')
    (root / '.release').write_text('a' * 32 + '\n')
    (bundle / 'scripts/deployment').mkdir(parents=True)
    (bundle / 'scripts/deployment/rollout.py').write_text(
        'from pathlib import Path\nimport sys\nPath(sys.argv[1], "applied").touch()\n')
    crontab = tools / 'crontab'
    crontab.write_text('#!/bin/sh\nexit 0\n')
    crontab.chmod(0o755)
    monkeypatch.setenv('PATH', str(tools) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('DEPLOY_LOCK_TIMEOUT', '2')
    return root, ['bash', str(root / 'deploy.sh')]


@pytest.mark.parametrize('lock', [fcntl.LOCK_SH, fcntl.LOCK_EX])
def test_deploy_waits_for_running_job_then_applies_release(release, lock):
    root, command = release
    process = None
    try:
        with (root / '.deployment.lock').open('w') as held:
            fcntl.flock(held, lock)
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert select.select([process.stdout], [], [], 5)[0], 'Missing lock progress message'
            assert 'Waiting for deployment/maintenance lock' in process.stdout.readline()
            assert process.poll() is None
            assert not (root / 'applied').exists()
            fcntl.flock(held, fcntl.LOCK_UN)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stderr
        assert 'lock acquired' in stdout
        assert (root / 'applied').exists()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()


def test_busy_lock_timeout_has_explicit_diagnostic_without_starting_deploy(release, monkeypatch):
    root, command = release
    monkeypatch.setenv('DEPLOY_LOCK_TIMEOUT', '0')
    with (root / '.deployment.lock').open('w') as held:
        fcntl.flock(held, fcntl.LOCK_SH)
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    assert result.returncode == 75
    assert 'Could not acquire deployment/maintenance lock' in result.stderr
    assert not (root / 'applied').exists()


@pytest.mark.parametrize('timeout', ['-1', 'not-a-number'])
def test_invalid_lock_timeout_is_rejected_before_starting_deploy(release, monkeypatch, timeout):
    root, command = release
    monkeypatch.setenv('DEPLOY_LOCK_TIMEOUT', timeout)
    result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert 'DEPLOY_LOCK_TIMEOUT' in result.stderr
    assert not (root / 'applied').exists()


@pytest.mark.parametrize('release_id', ['b' * 32, 'release-test'])
def test_stable_entrypoint_selects_the_requested_incoming_bundle(release, release_id):
    root, command = release
    bundle = root / 'incoming' / release_id
    (bundle / 'scripts/deployment').mkdir(parents=True)
    (bundle / 'scripts/deployment/rollout.py').write_text(
        'from pathlib import Path\nimport sys\nPath(sys.argv[1], "selected").write_text(sys.argv[2])\n')
    (root / '.release').write_text(release_id + '\n')
    result = subprocess.run(command, cwd=root.parent,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert (root / 'selected').read_text() == str(bundle)
    assert not (root / 'applied').exists()


def test_missing_bundle_is_rejected_before_deployment(release):
    root, command = release
    (root / '.release').write_text('c' * 32 + '\n')
    result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert 'Release bundle is unavailable' in result.stderr
    assert not (root / 'applied').exists()


def test_first_maintenance_installs_static_slots_and_records_green(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys

    source = ROOT / 'scripts/deployment'
    monkeypatch.syspath_prepend(str(source))
    spec = importlib.util.spec_from_file_location('release_bundle', source / 'release-bundle.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    from _rollout.checksum import get_checksum

    root, tools = tmp_path / 'root', tmp_path / 'tools'
    root.mkdir()
    tools.mkdir()
    bundle = root / 'incoming' / ('a' * 32)
    bundle.parent.mkdir()
    shutil.copy2(ROOT / '.deploy/deploy.sh', root / 'deploy.sh')
    (root / '.release').write_text('a' * 32 + '\n')
    builder._copy_release_files(bundle)
    manifest = {'mode': 'maintenance', 'release_id': 'a' * 32, 'checksum': get_checksum(bundle)}
    (root / '.release_manifest').write_text(json.dumps(manifest))
    for name in ('.env', '.env.local'):
        (root / name).touch()
    log = tmp_path / 'docker-calls.jsonl'
    docker = tools / 'docker'
    docker.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
args = sys.argv[1:]
with open(os.environ['DOCKER_CALLS'], 'a') as output:
    output.write(json.dumps(args) + '\\n')
if 'ps' in args and '--services' in args:
    print('postgres\\nnginx\\nclio\\nclio-worker')
elif 'sh' in args and 'pg_dumpall' in args[-1]:
    print('database dump')
if 'alembic' in args:
    import fcntl
    from pathlib import Path
    with open(Path(os.environ['DEPLOY_TEST_ROOT']) / '.deployment.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise AssertionError('Deployment must retain its exclusive lock during migrations')
''')
    docker.chmod(0o755)
    crontab = tools / 'crontab'
    crontab.write_text('#!/bin/sh\ncat >/dev/null\n')
    crontab.chmod(0o755)
    monkeypatch.setenv('PATH', str(tools) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('DOCKER_CALLS', str(log))
    monkeypatch.setenv('DEPLOY_TEST_ROOT', str(root))
    result = subprocess.run(['bash', str(root / 'deploy.sh')],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / '.active-color').read_text() == 'green\n'
    assert not (root / '.rolling-compose.json').exists()
    assert not (root / '.rolling-state.json').exists()
    assert not (root / '.maintenance-recovery.json').exists()
    assert len(list((root / 'backups').glob('*.sql.gz'))) == 1
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    migrations = [args[-4:] for args in calls if 'alembic' in args]
    assert migrations == [[domain, 'alembic', 'upgrade', 'head']
                          for domain in ('api', 'clio', 'prophet', 'intelligence')]
    starts = [args[-1] for args in calls if '--force-recreate' in args and args[-1].endswith('-green')]
    assert set(starts) == {'clio-green', 'api-green', 'prophet-api-green', 'intelligence-api-green', 'frontend-green'}
    assert not any('.rolling-compose.json' in argument for args in calls for argument in args)
    with (root / '.deployment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
