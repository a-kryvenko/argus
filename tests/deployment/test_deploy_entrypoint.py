"""Exercise the real shell lock without SSH, Docker or production access."""
import fcntl
import os
from pathlib import Path
import select
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def release(tmp_path, monkeypatch):
    root, bundle, tools = (tmp_path / name for name in ('root', 'bundle', 'tools'))
    for path in (root, bundle, tools):
        path.mkdir()
    (root / '.rolling-state.json').write_text('{}')
    shutil.copy2(ROOT / 'scripts/deployment/deploy.sh', bundle / 'deploy.sh')
    (bundle / 'rolling.py').write_text(
        'from pathlib import Path\nimport sys\nPath(sys.argv[1], "applied").touch()\n')
    crontab = tools / 'crontab'
    crontab.write_text('#!/bin/sh\nexit 0\n')
    crontab.chmod(0o755)
    monkeypatch.setenv('PATH', str(tools) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('DEPLOY_LOCK_TIMEOUT', '2')
    return root, ['bash', str(bundle / 'deploy.sh'), str(root)]


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
