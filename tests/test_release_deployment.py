"""Release planning and host orchestration tests; never contact Docker or production."""
import importlib.util
import gzip
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


def test_deployment_waits_for_validation_while_images_build_concurrently():
    import yaml

    jobs = yaml.safe_load((ROOT / '.github/workflows/deploy.yml').read_text())['jobs']
    assert jobs['images']['needs'] == 'plan'
    assert set(jobs['deploy']['needs']) == {'plan', 'images', 'domain-storage'}
    assert 'if' not in jobs['deploy']  # Default success gate must remain in force.


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
    files = {'.gitignore': 'packages/forecast-core/\npackages/intelligence-core/\n', '.dockerignore': '**/.venv\n', 'docs/example.md': 'docs',
             'packages/common/src/common/shared.py': 'shared',
             'packages/forecast/src/forecast/example.py': 'forecast', 'apps/web/app/page.tsx': 'page',
             'apps/prophet/src/argus_prophet/worker.py': 'worker',
             'apps/intelligence/src/argus_intelligence/cli.py': 'cli',
             'apps/clio/src/clio/migrations/versions/test.py': 'migration',
             'configs/project.yaml': 'config'}
    files.update({f'.deploy/{name}.Dockerfile': 'FROM scratch' for name in release.IMAGES})
    tracked_repo(tmp_path, files)
    tracked_repo(tmp_path / 'packages/forecast-core', {'src/forecast_core/api.py': 'private'})
    tracked_repo(tmp_path / 'packages/intelligence-core', {'src/intelligence_core/api.py': 'private intelligence'})
    return tmp_path


def changed_components(before, after):
    return {name for name in before['components'] if before['components'][name] != after['components'][name]}


@pytest.mark.parametrize('path,expected', [
    ('apps/intelligence/src/argus_intelligence/cli.py', {'intelligence'}),
    ('packages/forecast/src/forecast/example.py', {'prophet', 'proswin'}),
    ('docs/example.md', set()), ('apps/web/app/page.tsx', {'frontend'}),
    ('packages/common/src/common/shared.py', {'api', 'clio', 'prophet', 'proswin', 'intelligence'}),
    ('packages/forecast-core/src/forecast_core/api.py', {'prophet'}),
    ('packages/intelligence-core/src/intelligence_core/api.py', {'intelligence'}),
    ('apps/prophet/src/argus_prophet/worker.py', {'prophet', 'proswin'}), ('configs/project.yaml', set()),
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
    (source / 'apps/clio/src/clio/migrations/versions/test.py').write_text('new migration')
    after = release.plan(source)
    assert changed_components(before, after) == {'clio'}
    assert {name for name in before['migrations'] if before['migrations'][name] != after['migrations'][name]} == {'clio'}


@pytest.mark.parametrize('domain,command', [('prophet', 'status'), ('intelligence', 'check')])
def test_operator_wrapper_selects_project_and_forwards_arguments(tmp_path, domain, command):
    root = tmp_path / 'host with spaces'
    (root / 'scripts').mkdir(parents=True)
    (root / '.argus-mode').write_text('prod\n')
    wrapper = root / 'scripts/run'
    wrapper.write_bytes((ROOT / 'scripts/run').read_bytes())
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
                if source == 'packages/forecast-core' and component == 'prophet':
                    continue
                if source == 'packages/intelligence-core' and component == 'intelligence':
                    continue
                assert any(source == prefix or source.startswith(prefix + '/') for prefix in prefixes), (component, source)




@pytest.mark.parametrize('migration,failure', [(None, None), ('clio', None), ('clio', 'migration'),
                                              ('clio', 'backup'), ('intelligence', None)])
@pytest.mark.parametrize('legacy_clio', [False, True])
def test_shell_deployment_and_success_checkpoint(tmp_path, migration, failure, legacy_clio):
    import os
    import shutil
    fail = failure is not None
    root, bundle, bin_dir = [tmp_path / name for name in ('host', 'bundle', 'tools')]
    for directory in (root, bundle, bin_dir):
        directory.mkdir()
    backup_dir = root / 'backups'
    backup_dir.mkdir()
    old_backups = []
    for day in range(1, 10):
        path = backup_dir / f'pre-migration-202601{day:02d}T000000.sql'
        path.write_text(f'old dump {day}')
        old_backups.append(path)
    (backup_dir / 'manual.sql').write_text('manual backup')
    (backup_dir / 'pre-migration-20260110T000000.sql.gz.partial').write_text('incomplete')
    (backup_dir / 'database-transfer').mkdir()
    (backup_dir / 'database-transfer' / 'database.sql').write_text('transfer backup')
    shutil.copy2(ROOT / 'scripts/deployment/deploy.sh', bundle / 'deploy.sh')
    shutil.copy2(ROOT / 'argus', bundle / 'argus')
    (bundle / 'scripts').mkdir()
    for name in ('run', 'logs'):
        shutil.copy2(ROOT / 'scripts' / name, bundle / 'scripts' / name)
    for name in ('configs', 'nginx', 'alloy'):
        (bundle / name).mkdir()
    fingerprints = 'intelligence same\napi same\nclio same\nprophet same\nconfigs same\nnginx same\nalloy same\n'
    (root / '.release-fingerprints.tsv').write_text(fingerprints)
    (bundle / 'fingerprints.tsv').write_text(fingerprints.replace(f'{migration} same', f'{migration} changed') if migration else fingerprints)
    for name in ('images.env', 'docker-compose.yml', 'release.json'):
        (bundle / name).write_text('{}')
    log = tmp_path / 'calls'
    docker = bin_dir / 'docker'
    docker.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$CALL_LOG"
if [[ "$*" == *'config --services' ]]; then printf '%s\\n' $INSTALLED_SERVICES; fi
if [[ "$FAIL_MIGRATION" == 1 && "$*" == *'run --rm --no-deps clio-migrate'* ]]; then exit 1; fi
if [[ "$*" == *'pg_dumpall'* ]]; then
    printf '%s\\n' 'database dump'
    if [[ "$FAIL_BACKUP" == 1 ]]; then exit 1; fi
fi
''')
    docker.chmod(0o755)
    rsync = bin_dir / 'rsync'
    rsync.write_text('#!/bin/sh\nexit 0\n')
    rsync.chmod(0o755)
    result = subprocess.run(['bash', str(bundle / 'deploy.sh'), str(root)],
                            env={**os.environ, 'PATH': str(bin_dir) + ':' + os.environ['PATH'],
                                 'CALL_LOG': str(log), 'FAIL_MIGRATION': str(int(failure == 'migration')),
                                 'FAIL_BACKUP': str(int(failure == 'backup')),
                                 'INSTALLED_SERVICES': 'api clio prophet prophet-api intelligence ' + (
                                     'solar-wind geomagnetic clio-refresh clio-aggregate' if legacy_clio else 'clio-worker')},
                            capture_output=True, text=True)
    assert (result.returncode != 0) == fail, result.stderr
    calls = log.read_text().splitlines()
    prune = calls.index('image prune -a -f --filter until=168h')
    pull = next(i for i, call in enumerate(calls) if 'pull --policy missing' in call)
    assert prune < pull
    assert not any('python' in call or '--force-recreate' in call or 'db-bootstrap' in call or '-provision' in call for call in calls)
    if migration:
        stop = next(i for i, call in enumerate(calls) if ' stop ' in call)
        backup = next(i for i, call in enumerate(calls) if 'pg_dumpall' in call)
        assert stop < backup
        if failure == 'backup':
            assert not any('run --rm --no-deps' in call for call in calls)
        else:
            migrate = next(i for i, call in enumerate(calls) if f'run --rm --no-deps {migration}-migrate' in call)
            assert backup < migrate
        assert '--dbname=' not in calls[backup]  # Back up all domain databases and owners.
        new_backups = [p for p in backup_dir.glob('pre-migration-*.sql.gz') if not p.name.startswith('pre-migration-202601')]
        assert len(new_backups) == (0 if failure == 'backup' else 1)
        if new_backups:
            assert new_backups[0].stat().st_mode & 0o777 == 0o600
            assert gzip.decompress(new_backups[0].read_bytes()) == b'database dump\n'
        expected = ['intelligence'] if migration == 'intelligence' else ['clio']
        if migration == 'clio' and not legacy_clio:
            expected.append('clio-worker')
        if legacy_clio:
            expected.extend(['solar-wind', 'geomagnetic', 'clio-refresh', 'clio-aggregate'])
        assert calls[stop].endswith('stop ' + ' '.join(expected))
    else:
        assert not any('pg_dumpall' in call or '-migrate' in call for call in calls)
        assert any(' stop ' in call for call in calls) == legacy_clio
    retired = 'solar-wind geomagnetic clio-refresh clio-aggregate'
    if legacy_clio:
        drain = next(i for i, call in enumerate(calls) if ' stop ' in call)
        remove = next(i for i, call in enumerate(calls) if call.endswith('rm -f ' + retired))
        assert drain < remove
        if not fail:
            start = next(i for i, call in enumerate(calls) if 'up -d --wait --wait-timeout' in call)
            assert remove < start
    else:
        assert not any(' rm ' in call for call in calls)
    assert (root / '.release-fingerprints.tsv').read_text() == (fingerprints if fail else (bundle / 'fingerprints.tsv').read_text())
    assert not fail or not any('nginx -s reload' in call for call in calls)
    assert (backup_dir / 'manual.sql').read_text() == 'manual backup'
    assert (backup_dir / 'pre-migration-20260110T000000.sql.gz.partial').read_text() == 'incomplete'
    assert (backup_dir / 'database-transfer' / 'database.sql').read_text() == 'transfer backup'
    if migration and not fail:
        assert not list(backup_dir.glob('pre-migration-*.sql'))
        assert len(list(backup_dir.glob('pre-migration-*.sql.gz'))) == 7
        for index, path in enumerate(old_backups):
            compressed = path.with_suffix('.sql.gz')
            assert compressed.exists() == (index >= 3)
            if compressed.exists():
                assert gzip.decompress(compressed.read_bytes()).decode() == f'old dump {index + 1}'
    else:
        assert all(path.exists() for path in old_backups)
    if failure == 'backup':
        assert len(list(backup_dir.glob('*.partial'))) == 2
        assert not (root / '.argus-mode').exists()
        return
    assert (root / '.argus-mode').read_text() == 'prod\n'
    assert (root / 'argus').read_bytes() == (ROOT / 'argus').read_bytes()
    assert (root / 'bin/argus').resolve() == root / 'argus'
    assert (root / 'scripts/run').is_file()
    assert (root / 'scripts/logs').is_file()


def test_plan_pins_both_private_repositories(source):
    plan = release.plan(source)
    for package, key in [('forecast-core', 'private_commit'), ('intelligence-core', 'intelligence_core_commit')]:
        expected = subprocess.check_output(['git', '-C', str(source / 'packages' / package), 'rev-parse', 'HEAD'], text=True).strip()
        assert plan[key] == expected


def test_bundle_includes_only_shared_command_scripts(tmp_path):
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps({'components': {}, 'migrations': {}, 'config_hash': 'test'}))
    destination = tmp_path / 'release'
    release.bundle(ROOT, plan_path, tmp_path, destination)
    assert {p.name for p in (destination / 'scripts').iterdir()} == {'run', 'logs'}
    for name in ('run', 'logs'):
        assert (destination / 'scripts' / name).read_bytes() == (ROOT / 'scripts' / name).read_bytes()
