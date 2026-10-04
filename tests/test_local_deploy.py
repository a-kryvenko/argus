"""Release checks must finish before any upload or Git mutation."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKS = [
    'pnpm --filter web lint',
    'pnpm --filter web check-types',
    'pnpm --filter web test',
    'pnpm --filter web test:dashboard',
    'test-python',
]


@pytest.mark.parametrize('failure', [None, *CHECKS])
def test_checks_gate_release_and_tests_only_never_publishes(tmp_path, failure):
    shutil.copy2(ROOT / 'deploy.sh', tmp_path / 'deploy.sh')
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    (tmp_path / 'scripts').mkdir()
    log = tmp_path / 'commands'
    stub = f'#!{sys.executable}\n' + '''import os
from pathlib import Path
import sys
command = ' '.join([Path(sys.argv[0]).name, *sys.argv[1:]])
with open(os.environ['CHECK_LOG'], 'a') as log:
    log.write(command + '\\n')
if Path(sys.argv[0]).name in ('git', 'rsync'):
    raise SystemExit(99)
raise SystemExit(17 if command == os.getenv('FAIL_CHECK') else 0)
'''
    for file in [*(binaries / name for name in ('pnpm', 'git', 'rsync')),
                 tmp_path / 'scripts/test-python']:
        file.write_text(stub)
        file.chmod(0o755)
    result = subprocess.run(['bash', str(tmp_path / 'deploy.sh'), *(['patch', '-m', 'Release'] if failure else [])],
                            env={**os.environ, 'PATH': str(binaries) + os.pathsep + os.environ['PATH'],
                                 'CHECK_LOG': str(log), 'FAIL_CHECK': failure or ''},
                            capture_output=True, text=True)
    assert result.returncode == (17 if failure else 0), result.stdout + result.stderr
    expected = CHECKS[:CHECKS.index(failure)+1] if failure else CHECKS
    assert log.read_text().splitlines() == expected


@pytest.mark.parametrize('args', [['-t'], ['patch'], ['minor'], ['major'], ['-m'], ['invalid']])
def test_invalid_arguments_fail_before_checks(tmp_path, args):
    shutil.copy2(ROOT / 'deploy.sh', tmp_path / 'deploy.sh')
    result = subprocess.run(['bash', str(tmp_path / 'deploy.sh'), *args],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert 'Usage:' in result.stderr
    assert 'Running web' not in result.stdout


@pytest.mark.parametrize('bump,tag', [(None, None), ('patch', 'v1.2.4'),
                                     ('minor', 'v1.3.0'), ('major', 'v2.0.0')])
def test_commit_and_release_modes(tmp_path, bump, tag):
    shutil.copy2(ROOT / 'deploy.sh', tmp_path / 'deploy.sh')
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    (tmp_path / 'scripts').mkdir()
    log = tmp_path / 'commands'
    stub = f'#!{sys.executable}\n' + '''import os
from pathlib import Path
import sys
command = ' '.join([Path(sys.argv[0]).name, *sys.argv[1:]])
with open(os.environ['CHECK_LOG'], 'a') as log:
    log.write(command + '\\n')
if command == 'git tag --sort=-v:refname':
    print('v1.2.3')
if 'status --porcelain' in command:
    print(' M example')
'''
    for file in [*(binaries / name for name in ('pnpm', 'git', 'rsync')),
                 tmp_path / 'scripts/test-python']:
        file.write_text(stub)
        file.chmod(0o755)
    args = ([bump] if bump else []) + ['-m', 'Release message']
    result = subprocess.run(['bash', str(tmp_path / 'deploy.sh'), *args],
                            env={**os.environ, 'PATH': str(binaries) + os.pathsep + os.environ['PATH'],
                                 'CHECK_LOG': str(log)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    commands = log.read_text().splitlines()
    assert commands[:len(CHECKS)] == CHECKS
    for repo in ('notebooks', 'packages/forecast-core', 'packages/intelligence-core', '.'):
        assert f'git -C {repo} commit -m Release message' in commands
        assert f'git -C {repo} push origin HEAD' in commands
    uploads = [command for command in commands if command.startswith('rsync ')]
    if tag:
        assert len(uploads) == 2
        assert commands[-2:] == [f'git tag -a {tag} -m Release message', f'git push origin {tag}']
    else:
        assert not uploads
        assert not any(command.startswith(('git tag', 'git push origin v')) for command in commands)
