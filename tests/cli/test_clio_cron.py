"""Host cron job locks and exit status."""
import fcntl
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def host(tmp_path):
    root = tmp_path / 'project with spaces'
    (root / 'scripts').mkdir(parents=True)
    shutil.copy2(ROOT / 'scripts/cron-run', root / 'scripts/cron-run')
    (root / 'argus').write_text('#!/bin/sh\nprintf "%s\\n" "$@"\nexit "${JOB_EXIT:-0}"\n')
    (root / 'argus').chmod(0o755)
    def run(*args, **kwargs):
        return subprocess.run([str(root / 'scripts/cron-run'), 'clio', *args], cwd=tmp_path,
                              env={**os.environ, **kwargs.pop('env', {})}, capture_output=True, text=True,
                              timeout=10, **kwargs)
    return root, run


def test_same_job_is_skipped_and_another_job_can_run(host):
    root, run = host
    (root / '.cron-locks').mkdir()
    with (root / '.cron-locks/clio-normalize.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run('normalize', 'normalize').returncode == 0
        assert (root / 'log/clio/normalize.log').read_text() == ''
        assert run('live', 'collect', 'dst').returncode == 0
        assert (root / 'log/clio/live.log').read_text().splitlines() == ['clio', 'collect', 'dst']
    assert run('normalize', 'normalize').returncode == 0
    assert (root / 'log/clio/normalize.log').read_text().splitlines() == ['clio', 'normalize']


def test_command_failures_are_logged_and_exit_status_is_preserved(host):
    root, run = host
    result = run('normalize', 'normalize', env={'JOB_EXIT': '7'})
    assert result.returncode == 7
    assert (root / 'log/clio/normalize.log').read_text().splitlines() == ['clio', 'normalize']


def test_prophet_jobs_wait_for_other_jobs_without_overlapping(host):
    import time
    root, _ = host
    (root / 'argus').write_text('''#!/bin/bash
set -eu
root=$(dirname -- "$0")
mkdir "$root/active" || exit 88
touch "$root/started-$2"
read -r release
rmdir "$root/active"
''')
    def wait_for(path):
        deadline = time.monotonic()+5
        while not path.exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
    def start(job):
        return subprocess.Popen([str(root / 'scripts/cron-run'), 'prophet', job, job],
                                stdin=subprocess.PIPE, text=True)
    first, second = start('first'), None
    try:
        wait_for(root / 'started-first')
        second = start('second')
        wait_for(root / '.cron-locks/prophet-second.lock')
        assert second.poll() is None and not (root / 'started-second').exists()
        first.communicate('\n', timeout=5)
        assert first.returncode == 0
        wait_for(root / 'started-second')
        second.communicate('\n', timeout=5)
        assert second.returncode == 0
    finally:
        for process in (first, second):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
