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


def test_registry_lookup_pins_digest_without_pulling_layers(monkeypatch):
    digest = 'sha256:' + 'a' * 64
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, digest + '\n', '')
    monkeypatch.setattr(release.subprocess, 'run', run)
    assert release.resolve_image('ghcr.io/example/clio', 'src-test') == 'ghcr.io/example/clio@' + digest
    assert calls == [['docker', 'buildx', 'imagetools', 'inspect', 'ghcr.io/example/clio:src-test',
                      '--format', '{{.Manifest.Digest}}']]


@pytest.mark.parametrize('error', [
    'ERROR: ghcr.io/example/clio:src-test: not found',
    'ERROR: manifest unknown',
    'ERROR: name unknown: repository name not known to registry',
])
def test_missing_registry_image_needs_build(monkeypatch, error):
    monkeypatch.setattr(release.subprocess, 'run',
                        lambda *a, **kw: subprocess.CompletedProcess(a, 1, '', error))
    assert release.resolve_image('ghcr.io/example/clio', 'src-test') is None


@pytest.mark.parametrize('error', [
    'ERROR: unauthorized: authentication required',
    'ERROR: denied: permission denied',
    'ERROR: unexpected status from HEAD request: 404 Not Found',
    'ERROR: failed to authorize: token endpoint: not found',
    'ERROR: dial tcp: network is unreachable',
])
def test_registry_failures_stop_planning(monkeypatch, error):
    monkeypatch.setattr(release.subprocess, 'run',
                        lambda *a, **kw: subprocess.CompletedProcess(a, 1, '', error))
    with pytest.raises(RuntimeError, match='ERROR:'):
        release.resolve_image('ghcr.io/example/clio', 'src-test')


def test_invalid_registry_digest_stops_planning(monkeypatch):
    monkeypatch.setattr(release.subprocess, 'run',
                        lambda *a, **kw: subprocess.CompletedProcess(a, 0, 'invalid', ''))
    with pytest.raises(RuntimeError, match='invalid image digest'):
        release.resolve_image('ghcr.io/example/clio', 'src-test')


def test_registry_timeout_stops_planning(monkeypatch):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(release.subprocess, 'run', timeout)
    with pytest.raises(RuntimeError, match='timed out'):
        release.resolve_image('ghcr.io/example/clio', 'src-test')


def test_deployment_waits_for_validation_while_images_build_concurrently():
    import yaml

    jobs = yaml.safe_load((ROOT / '.github/workflows/deploy.yml').read_text())['jobs']
    assert jobs['images']['needs'] == 'plan'
    assert set(jobs['deploy']['needs']) == {'plan', 'validate', 'images'}
    assert jobs['validate']['needs'] == 'plan'
    assert jobs['images']['strategy']['matrix'] == '${{ fromJSON(needs.plan.outputs.build_matrix) }}'


def test_image_jobs_checkout_private_docker_inputs():
    import re
    import shlex
    import yaml

    job = yaml.safe_load((ROOT / '.github/workflows/deploy.yml').read_text())['jobs']['images']
    steps = job['steps']
    build_index = next(i for i, step in enumerate(steps) if step.get('uses', '').startswith('docker/build-push-action@'))
    for component in release.IMAGES:
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
    files.update({f'apps/{domain}/alembic.ini': f'[alembic]\nscript_location = {domain}:db/migrations\n'
                  for domain in release.MIGRATIONS})
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
    *((f'apps/{domain}/alembic.ini', {domain}) for domain in release.MIGRATIONS),
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
    if path.endswith('/alembic.ini'):
        assert {name for name in before['migrations']
                if before['migrations'][name] != after['migrations'][name]} == expected


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


@pytest.mark.parametrize('plan,validate,images,builds,cancelled,allowed', [
    ('success', 'success', 'success', 'true', False, True),
    ('success', 'success', 'skipped', 'false', False, True),
    ('success', 'success', 'skipped', 'true', False, False),
    ('success', 'success', 'failure', 'true', False, False),
    ('success', 'success', 'cancelled', 'true', False, False),
    ('success', 'failure', 'success', 'true', False, False),
    ('failure', 'skipped', 'skipped', '', False, False),
    ('success', 'success', 'success', 'true', True, False),
])
def test_deploy_gate_requires_success_even_when_builds_are_skipped(plan, validate, images, builds, cancelled, allowed):
    import yaml
    expression = yaml.safe_load((ROOT / '.github/workflows/deploy.yml').read_text())['jobs']['deploy']['if']
    expression = expression.strip().removeprefix('${{').removesuffix('}}').strip()
    replacements = {
        'needs.plan.result': plan, 'needs.validate.result': validate,
        'needs.images.result': images,
        'needs.plan.outputs.has_builds': builds,
    }
    for name, value in replacements.items():
        expression = expression.replace(name, repr(value))
    expression = expression.replace('!cancelled()', repr(not cancelled)).replace('&&', 'and').replace('||', 'or')
    assert eval('(' + expression + ')', {'__builtins__': {}}) is allowed


@pytest.mark.parametrize('missing', [set(), {'clio'}, set(release.IMAGES)])
def test_resolve_builds_only_missing_images_and_bundles_cached_results(tmp_path, missing):
    manifest = {'components': {name: {'repository': 'ghcr.io/example/' + name, 'tag': 'src-test'}
                               for name in release.IMAGES}, 'migrations': {}, 'config_hash': 'test'}
    def lookup(repository, tag):
        return None if repository.rsplit('/', 1)[-1] in missing else repository + '@sha256:' + 'a'*64
    matrix = release.resolve_images(manifest, lookup)
    assert {item['component'] for item in matrix['include']} == missing
    assert set(manifest['cached_images']) == set(release.IMAGES) - missing
    images = tmp_path / 'images'
    images.mkdir()
    for name in missing:
        (images / (name + '.json')).write_text(json.dumps({
            'tag': 'src-test', 'image': 'ghcr.io/example/' + name + '@sha256:' + 'b'*64}))
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps(manifest))
    destination = tmp_path / 'bundle'
    release.bundle(ROOT, plan, images, destination)
    result = json.loads((destination / 'release.json').read_text())
    assert set(result['images']) == set(release.IMAGES)
    for name, image in result['images'].items():
        assert image.endswith(('b' if name in missing else 'a')*64)


@pytest.mark.parametrize('bad_result', [None, {'tag': 'wrong', 'image': 'ghcr.io/example/clio@sha256:' + 'a'*64}])
def test_bundle_rejects_missing_or_mismatched_built_image(tmp_path, bad_result):
    manifest = {'components': {'clio': {'repository': 'ghcr.io/example/clio', 'tag': 'src-test'}}}
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps(manifest))
    if bad_result:
        (tmp_path / 'clio.json').write_text(json.dumps(bad_result))
    with pytest.raises(ValueError, match='Missing image result|does not match'):
        release.bundle(ROOT, plan, tmp_path, tmp_path / 'bundle')


@pytest.mark.parametrize('path', [
    'packages/common/tests/test_example.py', 'packages/common/docs/example.md',
    'packages/forecast/tests/test_example.py', 'apps/web/tests/example.spec.ts', 'apps/web/playwright.config.ts',
    'packages/forecast-core/docs/example.md', 'packages/intelligence-core/tests/test_example.py',
])
def test_non_runtime_changes_do_not_rebuild_images(source, path):
    file = source / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text('original')
    repository = next((source / 'packages' / name for name in ('forecast-core', 'intelligence-core')
                       if file.is_relative_to(source / 'packages' / name)), source)
    subprocess.run(['git', '-C', str(repository), 'add', '.'], check=True)
    before = release.plan(source)
    file.write_text('changed')
    assert changed_components(before, release.plan(source)) == set()
