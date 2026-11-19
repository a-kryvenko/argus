"""Release checks must finish before any upload or Git mutation."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKS = [
    'pnpm --filter web lint',
    'pnpm --filter web check-types',
    'pnpm --filter web test',
    'pnpm --filter web test:dashboard',
    'test-python',
    'pnpm docs:architecture',
    *[f'docker build --file .docker/{component}.Dockerfile --tag argus-{component}:build-check '
      '--build-arg DEPLOYMENT_VERSION=build-check .'
      for component in ('frontend', 'api', 'clio', 'prophet', 'proswin', 'intelligence')],
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
    for file in [*(binaries / name for name in ('pnpm', 'docker', 'git', 'rsync')),
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


@pytest.mark.parametrize('args', [
    ['-t'], ['patch'], ['minor'], ['major'], ['-m'], ['invalid'],
    ['--maintenance'], ['-m', 'Message', '--maintenance'],
])
def test_invalid_arguments_fail_before_checks(tmp_path, args):
    shutil.copy2(ROOT / 'deploy.sh', tmp_path / 'deploy.sh')
    result = subprocess.run(['bash', str(tmp_path / 'deploy.sh'), *args],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert 'Usage:' in result.stderr
    assert 'Running web' not in result.stdout


@pytest.mark.parametrize('bump,tag,maintenance', [
    (None, None, False), ('patch', 'v1.2.4', False), ('patch', 'v1.2.4', True),
    ('minor', 'v1.3.0', False), ('major', 'v2.0.0', False),
    ('patch', 'upload-failure', False), ('patch', 'manifest-upload-failure', False),
])
@pytest.mark.parametrize('dirty_submodule_only', [False, True])
def test_commit_and_release_modes(tmp_path, bump, tag, maintenance, dirty_submodule_only):
    shutil.copy2(ROOT / 'deploy.sh', tmp_path / 'deploy.sh')
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/deployment').mkdir()
    artifact_id = 'a' * 64
    (tmp_path / 'scripts/deployment/release-bundle.py').write_text(
        'import json, sys\nfrom pathlib import Path\n'
        'assert sys.argv[1] == "--output"\n'
        f'assert sys.argv[3:] == {(["--maintenance"] if maintenance else [])!r}\n'
        'destination = Path(sys.argv[2])\n'
        'destination.mkdir()\n'
        '(destination / "deploy.sh").write_text("server entrypoint")\n'
        f'manifest = {{"release_id": {artifact_id!r}, "mode": {("maintenance" if maintenance else "rolling")!r}}}\n'
        '(destination.parent / ".release_manifest").write_text(json.dumps(manifest))\n'
        '(destination.parent / ".release").write_text(manifest["release_id"] + "\\n")\n'
        f'print({artifact_id!r})\n'
    )
    log = tmp_path / 'commands'
    stub = f'#!{sys.executable}\n' + '''import json, os
from pathlib import Path
import sys
command = ' '.join([Path(sys.argv[0]).name, *sys.argv[1:]])
with open(os.environ['CHECK_LOG'], 'a') as log:
    log.write(command + '\\n')
if command == 'git tag --sort=-v:refname':
    print('v1.2.3')
if 'status --porcelain' in command:
    print(' M example')
if 'diff --cached --quiet' in command:
    raise SystemExit(0 if sys.argv[2] == 'notebooks' and os.getenv('DIRTY_SUBMODULE_ONLY') == '1' else 1)
if command.startswith('rsync ') and os.getenv('FAIL_BUNDLE_UPLOAD') == '1':
    raise SystemExit(17)
if command.startswith('rsync ') and sys.argv[-1] == 'argus:/var/www/':
    entrypoint = Path(sys.argv[-4])
    marker = Path(sys.argv[-2])
    manifest = Path(sys.argv[-3])
    assert marker.name == '.release' and manifest.name == '.release_manifest'
    assert entrypoint.name == 'deploy.sh' and entrypoint.read_text() == 'server entrypoint'
    assert json.loads(manifest.read_text())['release_id'] == marker.read_text().strip()
    if os.getenv('FAIL_METADATA_UPLOAD') == '1':
        raise SystemExit(17)
'''
    for file in [*(binaries / name for name in ('pnpm', 'docker', 'git', 'rsync')),
                 tmp_path / 'scripts/test-python']:
        file.write_text(stub)
        file.chmod(0o755)
    args = ([bump] if bump else []) + ['-m', 'Release message']
    if maintenance:
        args.append('--maintenance')
    result = subprocess.run(['bash', str(tmp_path / 'deploy.sh'), *args],
                            env={**os.environ, 'PATH': str(binaries) + os.pathsep + os.environ['PATH'],
                                 'CHECK_LOG': str(log), 'DIRTY_SUBMODULE_ONLY': str(int(dirty_submodule_only)),
                                 'FAIL_BUNDLE_UPLOAD': str(int(tag == 'upload-failure')),
                                 'FAIL_METADATA_UPLOAD': str(int(tag == 'manifest-upload-failure'))},
                            capture_output=True, text=True)
    if tag in ('upload-failure', 'manifest-upload-failure'):
        assert result.returncode == 17, result.stdout + result.stderr
        commands = log.read_text().splitlines()
        if tag == 'upload-failure':
            assert commands[-1].endswith(f'release-bundle/ argus:/var/www/incoming/{artifact_id}/')
        else:
            assert commands[-1].endswith('.release argus:/var/www/')
        assert not any(command.startswith(('git tag -a ', 'git push ')) for command in commands)
        return
    assert result.returncode == 0, result.stdout + result.stderr
    commands = log.read_text().splitlines()
    assert commands[:len(CHECKS)] == CHECKS
    for repo in ('notebooks', 'packages/forecast-core', 'packages/intelligence-core', '.'):
        assert (f'git -C {repo} commit -m Release message' in commands) == (repo != 'notebooks' or not dirty_submodule_only)
        assert f'git -C {repo} push origin HEAD' in commands
    uploads = [command for command in commands if command.startswith('rsync ')]
    if tag:
        assert len(uploads) == 2
        assert uploads[0].endswith(f'release-bundle/ argus:/var/www/incoming/{artifact_id}/')
        assert uploads[1].endswith('.release argus:/var/www/')
        assert '.release_manifest ' in uploads[1]
        assert 'release-bundle/deploy.sh ' in uploads[1]
        assert commands[-2:] == [f'git tag -a {tag} -m Release message',
                                 f'git push origin {tag}']
    else:
        assert not uploads
        assert not any(command.startswith(('git tag', 'git push origin v')) for command in commands)
