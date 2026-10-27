"""Release planning and host orchestration tests; never contact Docker or production."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


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


def test_image_jobs_checkout_private_docker_inputs():
    import re
    import shlex
    import yaml

    job = yaml.safe_load((ROOT / '.github/workflows/deploy.yml').read_text())['jobs']['images']
    steps = job['steps']
    build_index = next(i for i, step in enumerate(steps) if step.get('uses', '').startswith('docker/build-push-action@'))
    for component in job['strategy']['matrix']['component']:
        dockerfile = (ROOT / '.deploy' / (component + '.Dockerfile')).read_text()
        for line in dockerfile.splitlines():
            if not line.startswith('COPY ') or line.startswith('COPY --from='):
                continue
            for source in shlex.split(line)[1:-1]:
                for package in ('forecast-core', 'intelligence-core'):
                    path = 'packages/' + package
                    if source != path and not source.startswith(path + '/'):
                        continue
                    checkout = next(step for step in steps[:build_index]
                                    if step.get('uses', '').startswith('actions/checkout@')
                                    and step.get('with', {}).get('path') == path)
                    components = re.findall(r"matrix\.component == '([^']+)'", checkout.get('if', ''))
                    assert component in components, f'{component} copies {source} but skips its private checkout'
                    assert checkout['with']['ref'].startswith('${{ needs.plan.outputs.')


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
             'packages/forecast/src/forecast/example.py': 'forecast', 'apps/web/src/app/page.tsx': 'page',
             'apps/prophet/src/prophet/commands/generation.py': 'worker',
             'apps/intelligence/src/intelligence/cli.py': 'cli',
             'apps/clio/src/clio/db/migrations/versions/test.py': 'migration',
             'apps/api/src/api/db/migrations/versions/test.py': 'API migration',
             'configs/project.yaml': 'config'}
    files.update({f'.deploy/{name}.Dockerfile': 'FROM scratch' for name in release.IMAGES})
    tracked_repo(tmp_path, files)
    tracked_repo(tmp_path / 'packages/forecast-core', {'src/forecast_core/api.py': 'private',
        'src/forecast_core/models/proswin.py': 'neural runtime',
        'src/forecast_core/solar_events/bundled_models/argus-cme-speed-v2.joblib': 'private artifact'})
    tracked_repo(tmp_path / 'packages/intelligence-core', {'src/intelligence_core/api.py': 'private intelligence'})
    return tmp_path


def changed_components(before, after):
    return {name for name in before['components'] if before['components'][name] != after['components'][name]}


@pytest.mark.parametrize('path,expected', [
    ('apps/intelligence/src/intelligence/cli.py', {'intelligence'}),
    ('packages/forecast/src/forecast/example.py', {'prophet', 'proswin'}),
    ('docs/example.md', set()), ('apps/web/src/app/page.tsx', {'frontend'}),
    ('packages/common/src/common/shared.py', {'api', 'clio', 'prophet', 'proswin', 'intelligence'}),
    ('packages/forecast-core/src/forecast_core/api.py', {'prophet'}),
    ('packages/forecast-core/src/forecast_core/models/proswin.py', {'prophet', 'proswin'}),
    ('packages/forecast-core/src/forecast_core/solar_events/bundled_models/argus-cme-speed-v2.joblib', {'prophet'}),
    ('packages/intelligence-core/src/intelligence_core/api.py', {'intelligence'}),
    ('apps/prophet/src/prophet/commands/generation.py', {'prophet', 'proswin'}), ('configs/project.yaml', set()),
])
def test_only_actual_consumers_rebuild(source, path, expected):
    before = release.plan(source)
    (source / path).write_text('changed')
    after = release.plan(source)
    assert changed_components(before, after) == expected


def test_deleted_sources_and_explicit_rebuild_are_not_missed(source):
    before = release.plan(source)
    (source / 'apps/web/src/app/page.tsx').unlink()
    assert changed_components(before, release.plan(source)) == {'frontend'}
    assert changed_components(before, release.plan(source, rebuild='operator-request')) == set(release.IMAGES)


@pytest.mark.parametrize('failure', [None, 'corrupt', 'missing'])
def test_deploy_requires_the_CME_artifact_pinned_by_registry(source, failure):
    import hashlib
    import yaml
    artifact = source / 'packages/forecast-core/src/forecast_core/solar_events/bundled_models/argus-cme-speed-v2.joblib'
    identity = {'model': 'argus-cme-speed-v2', 'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}
    (source / 'configs/models_registry.yaml').write_text(yaml.safe_dump({'models': {'solar_event_speed': identity}}))
    if failure == 'corrupt':
        artifact.write_bytes(b'wrong model')
    elif failure == 'missing':
        artifact.unlink()
    if failure:
        with pytest.raises(ValueError, match='Bundled CME model'):
            release.plan(source)
    else:
        assert release.plan(source)['cme_speed_model'] == identity


def test_migrations_are_fingerprinted_separately(source):
    before = release.plan(source)
    (source / 'apps/clio/src/clio/db/migrations/versions/test.py').write_text('new migration')
    after = release.plan(source)
    assert changed_components(before, after) == {'clio'}
    assert {name for name in before['migrations'] if before['migrations'][name] != after['migrations'][name]} == {'clio'}


@pytest.mark.parametrize('domain,command', [('prophet', 'generate'), ('intelligence', 'check')])
def test_operator_wrapper_selects_project_and_forwards_arguments(tmp_path, domain, command):
    root = tmp_path / 'host with spaces'
    (root / 'scripts').mkdir(parents=True)
    (root / '.argus-mode').write_text('prod\n')
    wrapper = root / 'scripts/run'
    wrapper.write_bytes((ROOT / 'scripts/run').read_bytes())
    for name in ('.rolling-state.json', '.rolling-compose.json', '.release-compose.json'):
        (root / name).write_text('{}')
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
    assert str(root / '.release-compose.json') in args
    assert str(root / '.release-images.env') not in args
    expected = ['run', '--rm', '--no-deps', *(['-T'] if domain == 'prophet' else []), domain, domain, command, 'solar-wind-speed']
    assert args[-len(expected):] == expected


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
                if source == 'packages/forecast-core/src/forecast_core/models/proswin.py' and component == 'proswin':
                    continue
                if source == 'packages/intelligence-core' and component == 'intelligence':
                    continue
                assert any(source == prefix or source.startswith(prefix + '/') for prefix in prefixes), (component, source)




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
    assert not (destination / 'deploy-legacy.sh').exists()
    assert {p.name for p in (destination / 'scripts').iterdir()} == {'run', 'logs', 'cron-run'}
    for name in ('run', 'logs', 'cron-run'):
        assert (destination / 'scripts' / name).read_bytes() == (ROOT / 'scripts' / name).read_bytes()
    for service in ('clio', 'prophet', 'api', 'intelligence'):
        assert (destination / 'apps' / service / 'cronjobs.txt').read_bytes() == (ROOT / 'apps' / service / 'cronjobs.txt').read_bytes()
    assert (destination / '.deploy/cronjobs.txt').read_bytes() == (ROOT / '.deploy/cronjobs.txt').read_bytes()


def test_packaged_api_revision_changes_migration_fingerprint(source):
    before = release.plan(source)
    (source / 'apps/api/src/api/db/migrations/versions/test.py').write_text('new API revision')
    after = release.plan(source)
    assert before['migrations']['api'] != after['migrations']['api']
    assert changed_components(before, after) == {'api'}
