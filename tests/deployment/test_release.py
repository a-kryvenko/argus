"""Check current release bundles, image planning and CI handoff."""
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/deployment'))


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), ROOT / 'scripts/deployment' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan = load('release-plan')
builder = load('release-bundle')


@pytest.mark.parametrize('error,exists', [
    ('', True),
    ('ERROR: ghcr.io/example/clio:src-test: not found', False),
    ('ERROR: manifest unknown', False),
    ('ERROR: name unknown: repository name not known to registry', False),
    ('ERROR: unauthorized: authentication required', None),
    ('ERROR: unexpected status from HEAD request: 404 Not Found', None),
    ('ERROR: failed to authorize: token endpoint: not found', None),
    ('ERROR: dial tcp: network is unreachable', None),
])
def test_registry_lookup_distinguishes_cache_misses_from_failures(monkeypatch, error, exists):
    reference = 'ghcr.io/example/clio:src-test'
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, int(bool(error)), '', error)
    monkeypatch.setattr(plan.subprocess, 'run', run)
    if exists is None:
        with pytest.raises(RuntimeError, match='Cannot inspect'):
            plan.image_exists(reference)
    else:
        assert plan.image_exists(reference) is exists
    assert calls == [['docker', 'buildx', 'imagetools', 'inspect', reference]]


@pytest.mark.parametrize('path,consumers', [
    ('packages/common/src/common/shared.py', {'api', 'clio', 'prophet', 'proswin', 'intelligence'}),
    ('packages/forecast/src/forecast/shared.py', {'prophet', 'proswin'}),
    ('packages/forecast-core/src/forecast_core/models/proswin.py', {'prophet', 'proswin'}),
    ('packages/intelligence-core/src/intelligence_core/runtime.py', {'intelligence'}),
    ('apps/api/alembic.ini', {'api'}),
    ('apps/clio/src/clio/db/migrations/versions/test.py', {'clio'}),
    ('apps/clio/api-go/history.go', {'clio-api'}),
    ('.docker/clio-api.Dockerfile', {'clio-api'}),
    ('apps/prophet/api-go/releases.go', {'prophet-api'}),
    ('.docker/prophet-api.Dockerfile', {'prophet-api'}),
    ('apps/web/src/app/page.tsx', {'frontend'}),
    ('apps/web/tests/example.spec.ts', set()),
    ('configs/project.yaml', set()),
])
def test_image_changes_select_actual_consumers(tmp_path, monkeypatch, path, consumers):
    monkeypatch.setattr(plan, 'ROOT', tmp_path)
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    file.write_text('before')
    before, _ = plan.changed_services({})
    file.write_text('after')
    _, changed = plan.changed_services(before)
    assert set(changed) == consumers
    file.unlink()
    _, deleted = plan.changed_services(before)
    assert set(deleted) == consumers


@pytest.fixture
def release_source(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    shutil.copytree(ROOT / '.deploy', source / '.deploy')
    shutil.copytree(ROOT / 'scripts/deployment', source / 'scripts/deployment',
                    ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('docker-compose.yml', 'argus', 'scripts/run', 'scripts/logs'):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    (source / 'configs').mkdir()
    (source / 'configs/project.yaml').write_text('configuration')
    for name in ('models/model.bin', 'metrics/metrics.json', 'demo/current.json', 'prophet/worker-owned.json'):
        target = source / 'data' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('artifact')
    (source / '.env.local').write_text('operator secrets')
    monkeypatch.setattr(builder, 'ROOT', source)
    return source


@pytest.mark.parametrize('maintenance', [False, True])
def test_bundle_cli_includes_runtime_files_and_data_with_valid_checksum(tmp_path, monkeypatch, release_source, maintenance, capsys):
    destination = tmp_path / 'bundle'
    monkeypatch.setattr(sys, 'argv', ['release-bundle.py', '--output', str(destination),
                                     *(['--maintenance'] if maintenance else [])])
    builder.main()
    release_id = capsys.readouterr().out.strip()
    assert (destination.parent / '.release').read_text() == release_id + '\n'
    assert re.fullmatch('[0-9a-f]{32}', release_id)
    manifest = json.loads((destination.parent / '.release_manifest').read_text())
    assert not (destination / 'release.json').exists()
    assert not (destination / '.release_manifest').exists()
    assert manifest == {'mode': 'maintenance' if maintenance else 'rolling',
                        'release_id': release_id, 'checksum': builder.get_checksum(destination)}
    for name in ('models/model.bin', 'metrics/metrics.json', 'demo/current.json'):
        assert (destination / 'data' / name).read_text() == 'artifact'
    assert not (destination / 'data/prophet').exists()
    assert not (destination / '.env.local').exists()
    assert not (destination / 'images.env').exists()
    for name in ('docker-compose.yml', 'argus', 'scripts/run'):
        assert (destination / name).read_bytes() == (release_source / name).read_bytes()
    assert (destination / 'docker-compose.prod.yml').read_bytes() == (ROOT / '.deploy/docker-compose.yml').read_bytes()
    assert (destination / '.docker/maintenance/index.html').is_file()
    assert (destination / 'scripts/deployment/rollout.py').is_file()
    (destination / 'data/models/model.bin').write_text('corrupt')
    assert builder.get_checksum(destination) != manifest['checksum']


@pytest.mark.parametrize('missing', ['models', 'metrics', 'demo/current.json'])
def test_bundle_requires_all_data_artifacts(tmp_path, monkeypatch, release_source, missing):
    target = release_source / 'data' / missing
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    destination = tmp_path / 'bundle'
    monkeypatch.setattr(sys, 'argv', ['release-bundle.py', '--output', str(destination)])
    with pytest.raises(SystemExit, match='Missing data/'):
        builder.main()
    assert not destination.exists()


def test_checksum_includes_files_named_release_json(tmp_path):
    before = builder.get_checksum(tmp_path)
    (tmp_path / 'release.json').write_text('payload')
    assert builder.get_checksum(tmp_path) != before
