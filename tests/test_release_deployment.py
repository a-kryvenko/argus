"""Release planning and host orchestration tests; never contact Docker or production."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/deployment' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = load('release')


def tracked_repo(path, files):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '-q', str(path)], check=True)
    for name, content in files.items():
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)
    subprocess.run(['git', '-C', str(path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(path), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'fixture'], check=True)


@pytest.fixture
def source(tmp_path):
    files = {'.gitignore': 'packages/forecast-core/\n', '.dockerignore': '**/.venv\n', 'docs/example.md': 'docs',
             'packages/common/src/common/shared.py': 'shared', 'apps/web/app/page.tsx': 'page',
             'apps/prophet/src/argus_prophet/worker.py': 'worker',
             'apps/intelligence/src/argus_intelligence/cli.py': 'cli',
             'apps/clio/src/argus_clio/migrations/versions/test.py': 'migration',
             'configs/project.yaml': 'config'}
    files.update({f'.deploy/{name}.Dockerfile': 'FROM scratch' for name in release.IMAGES})
    tracked_repo(tmp_path, files)
    tracked_repo(tmp_path / 'packages/forecast-core', {'src/forecast_core/api.py': 'private'})
    return tmp_path


def changed_components(before, after):
    return {name for name in before['components'] if before['components'][name] != after['components'][name]}


@pytest.mark.parametrize('path,expected', [
    ('apps/intelligence/src/argus_intelligence/cli.py', {'intelligence'}),
    ('docs/example.md', set()), ('apps/web/app/page.tsx', {'frontend'}),
    ('packages/common/src/common/shared.py', {'api', 'clio', 'prophet', 'intelligence'}),
    ('packages/forecast-core/src/forecast_core/api.py', {'clio', 'prophet'}),
    ('apps/prophet/src/argus_prophet/worker.py', {'prophet'}), ('configs/project.yaml', set()),
])
def test_only_actual_consumers_rebuild(source, path, expected):
    before = release.plan(source)
    (source / path).write_text('changed')
    after = release.plan(source)
    assert changed_components(before, after) == expected


def test_deleted_sources_and_explicit_rebuild_are_not_missed(source):
    before = release.plan(source)
    (source / 'apps/web/app/page.tsx').unlink()
    assert changed_components(before, release.plan(source)) == {'frontend'}
    assert changed_components(before, release.plan(source, rebuild='operator-request')) == set(release.IMAGES)


def test_migrations_are_fingerprinted_separately(source):
    before = release.plan(source)
    (source / 'apps/clio/src/argus_clio/migrations/versions/test.py').write_text('new migration')
    after = release.plan(source)
    assert changed_components(before, after) == {'clio'}
    assert {name for name in before['migrations'] if before['migrations'][name] != after['migrations'][name]} == {'clio'}


@pytest.mark.parametrize('domain,command', [('prophet', 'status'), ('intelligence', 'check')])
def test_operator_wrapper_selects_project_and_forwards_arguments(tmp_path, domain, command):
    root = tmp_path / 'host with spaces'
    (root / 'bin').mkdir(parents=True)
    wrapper = root / 'bin/argus'
    wrapper.write_bytes((ROOT / 'scripts/deployment/argus').read_bytes())
    (root / '.release-images.env').write_text('images')
    fake_bin = tmp_path / 'tools'
    fake_bin.mkdir()
    docker = fake_bin / 'docker'
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    docker.chmod(0o755)
    import os
    result = subprocess.run(['bash', str(wrapper), domain, command, 'solar-wind-speed'],
                            cwd=tmp_path, env={**os.environ, 'PATH': str(fake_bin) + ':' + os.environ['PATH']},
                            check=True, capture_output=True, text=True)
    args = result.stdout.splitlines()
    assert args[args.index('--project-directory') + 1] == str(root)
    assert str(root / '.release-images.env') in args
    assert args[-7:] == ['run', '--rm', '--no-deps', domain, domain, command, 'solar-wind-speed']


def test_every_docker_copy_source_participates_in_image_identity():
    import shlex
    for component, prefixes in release.INPUTS.items():
        for line in (ROOT / '.deploy' / (component + '.Dockerfile')).read_text().splitlines():
            if not line.startswith('COPY '):
                continue
            words = shlex.split(line)
            if not words or words[0] != 'COPY' or words[1].startswith('--from='):
                continue
            for source in words[1:-1]:
                if source == 'packages/forecast-core' and component in ('clio', 'prophet'):
                    continue
                assert any(source == prefix or source.startswith(prefix + '/') for prefix in prefixes), (component, source)




@pytest.mark.parametrize('migration,fail', [(False, False), (True, False), (True, True)])
def test_shell_deployment_and_success_checkpoint(tmp_path, migration, fail):
    import os
    import shutil
    root, bundle, bin_dir = [tmp_path / name for name in ('host', 'bundle', 'tools')]
    for directory in (root, bundle, bin_dir):
        directory.mkdir()
    for name in ('deploy.sh', 'argus'):
        shutil.copy2(ROOT / 'scripts/deployment' / name, bundle / name)
    for name in ('configs', 'nginx', 'alloy'):
        (bundle / name).mkdir()
    fingerprints = 'api same\nclio same\nprophet same\nconfigs same\nnginx same\nalloy same\n'
    (root / '.release-fingerprints.tsv').write_text(fingerprints)
    (bundle / 'fingerprints.tsv').write_text(fingerprints.replace('clio same', 'clio changed') if migration else fingerprints)
    for name in ('images.env', 'docker-compose.yml', 'release.json'):
        (bundle / name).write_text('{}')
    log = tmp_path / 'calls'
    docker = bin_dir / 'docker'
    docker.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$CALL_LOG"
if [[ "$FAIL_MIGRATION" == 1 && "$*" == *'run --rm --no-deps clio-migrate'* ]]; then exit 1; fi
''')
    docker.chmod(0o755)
    rsync = bin_dir / 'rsync'
    rsync.write_text('#!/bin/sh\nexit 0\n')
    rsync.chmod(0o755)
    result = subprocess.run(['bash', str(bundle / 'deploy.sh'), str(root)],
                            env={**os.environ, 'PATH': str(bin_dir) + ':' + os.environ['PATH'],
                                 'CALL_LOG': str(log), 'FAIL_MIGRATION': str(int(fail))},
                            capture_output=True, text=True)
    assert (result.returncode != 0) == fail, result.stderr
    calls = log.read_text().splitlines()
    assert not any('python' in call or '--force-recreate' in call or 'db-bootstrap' in call for call in calls)
    if migration:
        stop = next(i for i, call in enumerate(calls) if ' stop ' in call)
        backup = next(i for i, call in enumerate(calls) if 'pg_dump' in call)
        migrate = next(i for i, call in enumerate(calls) if 'run --rm --no-deps clio-migrate' in call)
        assert stop < backup < migrate
        assert calls[stop].endswith('stop clio solar-wind geomagnetic clio-refresh clio-aggregate')
    else:
        assert not any(' stop ' in call or 'pg_dump' in call or '-migrate' in call for call in calls)
    assert (root / '.release-fingerprints.tsv').read_text() == (fingerprints if fail else (bundle / 'fingerprints.tsv').read_text())
    assert not fail or not any('nginx -s reload' in call for call in calls)
