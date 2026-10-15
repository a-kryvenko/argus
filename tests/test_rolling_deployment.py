"""Exercise failure ordering and durable recovery without production access."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/deployment'))
import rolling


def harness(tmp_path):
    deployment = rolling.Deployment.__new__(rolling.Deployment)
    deployment.root = tmp_path
    deployment.bundle = ROOT / 'scripts/deployment'
    deployment.runtime = tmp_path / '.rolling-compose.json'
    deployment.state_path = tmp_path / '.rolling-state.json'
    deployment.minimum_available = 0
    previous = {'image': 'example@sha256:old', 'networks': {'observations': {}}, 'environment': {}}
    desired = {**previous, 'image': 'example@sha256:new'}
    deployment.desired = {'services': {'clio': desired}, 'networks': {'observations': {}}}
    deployment.state = {'active': {'clio': {'slot': 'blue', 'spec': previous}}}
    deployment.credentials = {'DEPLOY_SMOKE_USERNAME': 'test', 'DEPLOY_SMOKE_PASSWORD': 'secret'}
    deployment.calls = []
    deployment.compose = lambda *args, **kwargs: deployment.calls.append(args) or ''
    deployment.smoke = lambda service, slot, routed=False: deployment.calls.append(('smoke', slot, routed))
    deployment.worker_pids = lambda service: set()
    deployment.save()
    return deployment


def test_replacement_starts_and_checks_before_switching_and_stopping(tmp_path):
    deployment = harness(tmp_path)
    deployment.replace('clio')
    calls = deployment.calls
    prepare = calls.index(('up', '-d', '--no-deps', '--wait', '--wait-timeout', '300', 'clio-green'))
    direct = calls.index(('smoke', 'clio-green', False))
    reload = calls.index(('exec', '-T', 'clio', 'nginx', '-s', 'reload'))
    routed = calls.index(('smoke', 'clio-green', True))
    stop = calls.index(('stop', 'clio-blue'))
    assert prepare < direct < reload < routed < stop
    assert deployment.state['active']['clio']['slot'] == 'green'
    assert deployment.state['active']['clio']['previous']['slot'] == 'blue'
    assert 'pending' not in json.loads(deployment.state_path.read_text())
    assert 'clio-green:8000' in (tmp_path / 'routes/clio/default.conf').read_text()
    assert deployment.state_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('failure', ['startup', 'direct-smoke', 'routed-smoke', 'stop-old'])
def test_failure_restores_previous_route_and_keeps_old_application(tmp_path, failure):
    deployment = harness(tmp_path)
    previous = copy.deepcopy(deployment.state['active']['clio'])
    original_compose, original_smoke = deployment.compose, deployment.smoke
    failed = False

    def compose(*args, **kwargs):
        nonlocal failed
        original_compose(*args, **kwargs)
        if not failed and ((failure == 'startup' and args[-1] == 'clio-green' and args[0] == 'up')
                           or (failure == 'stop-old' and args == ('stop', 'clio-blue'))):
            failed = True
            raise RuntimeError('injected failure')
        return ''

    def smoke(service, slot, routed=False):
        original_smoke(service, slot, routed)
        if slot == 'clio-green' and failure == ('routed-smoke' if routed else 'direct-smoke'):
            raise RuntimeError('injected failure')

    deployment.compose, deployment.smoke = compose, smoke
    with pytest.raises(RuntimeError, match='injected'):
        deployment.replace('clio')
    assert deployment.state['active']['clio'] == previous
    assert 'pending' not in deployment.state
    assert 'clio-blue:8000' in (tmp_path / 'routes/clio/default.conf').read_text()
    assert ('stop', 'clio-green') in deployment.calls
    assert ('up', '-d', '--no-deps', '--wait', '--wait-timeout', '300', 'clio-blue') in deployment.calls


def test_interrupted_switch_is_recovered_from_disk(tmp_path):
    deployment = harness(tmp_path)
    deployment.state['pending'] = {'service': 'clio', 'candidate': {'slot': 'green', 'spec': deployment.desired['services']['clio']}}
    deployment.save()
    deployment.route('clio', 'clio-green')
    deployment.state = json.loads(deployment.state_path.read_text())
    deployment.recover()
    assert 'clio-blue:8000' in (tmp_path / 'routes/clio/default.conf').read_text()
    assert 'pending' not in json.loads(deployment.state_path.read_text())


def test_unchanged_service_and_memory_gate_do_not_start_replacement(tmp_path):
    deployment = harness(tmp_path)
    deployment.desired['services']['clio'] = copy.deepcopy(deployment.state['active']['clio']['spec'])
    deployment.replace('clio')
    assert deployment.calls == []
    deployment.desired['services']['clio']['image'] = 'changed'
    deployment.minimum_available = 10**12
    with pytest.raises(RuntimeError, match='MiB available'):
        deployment.replace('clio')
    assert deployment.calls == []
    assert 'pending' not in deployment.state


def test_runtime_keeps_stopped_previous_slot_and_preserves_network_isolation(tmp_path):
    deployment = harness(tmp_path)
    deployment.replace('clio')
    config = json.loads(deployment.runtime.read_text())
    assert set(config['services']) == {'clio', 'clio-blue', 'clio-green'}
    assert config['services']['clio']['image'] == 'nginx:stable-alpine'
    assert config['services']['clio']['networks'] == {'observations': {}}
    assert 'environment' not in config['services']['clio']
    assert config['services']['clio-blue']['image'].endswith(':old')


def test_drain_waits_for_pre_reload_worker_even_before_signal_is_processed(tmp_path, monkeypatch):
    deployment = harness(tmp_path)
    generations = iter([{'100', '101'}, {'100', '102'}, {'102'}])
    deployment.worker_pids = lambda service: next(generations)
    sleeps = []
    monkeypatch.setattr(rolling.time, 'sleep', lambda duration: sleeps.append(duration))
    deployment.drain_router('clio', {'100'})
    assert len(sleeps) == 2


@pytest.mark.parametrize('arguments,expected', [
    (['logs', '--tail', '20', 'api'], ['logs', '--tail', '20', 'api-green']),
    (['exec', 'api', 'echo', 'api'], ['exec', 'api-green', 'echo', 'api']),
    (['run', '--name', 'api', 'api', 'echo', 'api'], ['run', '--name', 'api', 'api-green', 'echo', 'api']),
])
def test_operator_commands_select_active_slot_without_rewriting_command_arguments(tmp_path, arguments, expected):
    import os
    import subprocess
    root, tools = tmp_path / 'host', tmp_path / 'tools'
    (root / 'scripts').mkdir(parents=True)
    tools.mkdir()
    (root / 'scripts/run').write_bytes((ROOT / 'scripts/run').read_bytes())
    (root / '.argus-mode').write_text('prod\n')
    (root / '.rolling-compose.json').write_text('{}')
    (root / '.rolling-state.json').write_text(json.dumps({'active': {'api': {'slot': 'green'}}}))
    docker = tools / 'docker'
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    docker.chmod(0o755)
    result = subprocess.run(['bash', str(root / 'scripts/run'), 'compose', *arguments],
        env={**os.environ, 'PATH': str(tools) + ':' + os.environ['PATH']}, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines()[-len(expected):] == expected
    assert str(root / '.rolling-compose.json') in result.stdout


def test_new_migrations_require_exact_approval_before_mutation(tmp_path, monkeypatch):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    fingerprints = {key: 'same' for key in ('api', 'clio', 'prophet', 'intelligence', 'configs', 'nginx', 'alloy')}
    (tmp_path / '.release-fingerprints.tsv').write_text(''.join(f'{k} {v}\n' for k, v in fingerprints.items()))
    fingerprints['clio'] = 'new'
    (bundle / 'fingerprints.tsv').write_text(''.join(f'{k} {v}\n' for k, v in fingerprints.items()))
    (bundle / 'release.json').write_text('{}')
    (tmp_path / '.rolling-state.json').write_text('{"active":{}}')
    config = {'services': {'api': {'environment': {'DEPLOY_SMOKE_USERNAME': 'test', 'DEPLOY_SMOKE_PASSWORD': 'password'}}}}
    monkeypatch.setattr(rolling.Deployment, 'capture', staticmethod(lambda *args: json.dumps(config)))
    with pytest.raises(RuntimeError, match='must be approved'):
        rolling.Deployment(tmp_path, bundle)
    (bundle / 'release.json').write_text(json.dumps({'online_migrations': {'clio': 'new'}}))
    deployment = rolling.Deployment(tmp_path, bundle)
    assert deployment.migrations == ['clio']


def test_artifacts_are_verified_and_mounts_are_frozen(tmp_path):
    deployment = harness(tmp_path)
    deployment.fingerprints = {'configs': 'config-hash'}
    deployment.bundle = tmp_path / 'bundle'
    deployment.bundle.mkdir()
    (deployment.bundle / 'configs').mkdir()
    (deployment.bundle / 'configs/project.yaml').write_text('version: original\n')
    artifacts = tmp_path / 'staging'
    for name in ('models', 'metrics', 'demo'):
        (artifacts / name).mkdir(parents=True)
    (artifacts / 'models/model.bin').write_bytes(b'model')
    (artifacts / 'demo/current.json').write_text('{}')
    from release import artifact_digest
    digest = artifact_digest(artifacts)
    target = tmp_path / 'artifacts' / digest
    target.parent.mkdir()
    artifacts.rename(target)
    deployment.manifest = {'artifacts': {'id': digest}}
    deployment.desired['services']['clio']['volumes'] = [
        {'type': 'bind', 'source': str(tmp_path / name), 'target': '/var/www/' + name}
        for name in ('configs', 'data/models', 'data/metrics')]
    deployment.snapshot()
    mounts = deployment.desired['services']['clio']['volumes']
    assert mounts[0]['source'] == str(tmp_path / 'releases/config-hash/configs')
    assert mounts[1]['source'] == str(target / 'models')
    assert mounts[2]['source'] == str(target / 'metrics')
    assert (tmp_path / 'data/demo/current.json').read_text() == '{}'
    (target / 'models/model.bin').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='missing or corrupt'):
        deployment.snapshot()


def test_initial_transition_restores_legacy_service_on_routed_failure(tmp_path):
    deployment = harness(tmp_path)
    deployment.state['active']['clio']['slot'] = None
    original = deployment.smoke
    def smoke(service, slot, routed=False):
        original(service, slot, routed)
        if routed:
            raise RuntimeError('injected transition failure')
    deployment.smoke = smoke
    with pytest.raises(RuntimeError, match='injected'):
        deployment.replace('clio')
    assert deployment.state['active']['clio']['slot'] is None
    assert ('stop', 'clio-blue') in deployment.calls
    assert json.loads(deployment.runtime.read_text())['services']['clio']['image'].endswith(':old')


def test_host_application_checkpoints_release_and_never_starts_all_http_slots(tmp_path):
    deployment = harness(tmp_path)
    for service in rolling.HTTP:
        if service == 'clio':
            continue
        spec = {'image': 'unchanged', 'networks': {'observations': {}}, 'environment': {}}
        deployment.desired['services'][service] = copy.deepcopy(spec)
        deployment.state['active'][service] = {'slot': 'blue', 'spec': spec}
    for service in [*rolling.WORKERS, 'postgres', 'redis', 'nginx', 'alloy']:
        deployment.desired['services'][service] = {'image': 'infrastructure'}
    deployment.bundle = tmp_path / 'bundle'
    deployment.bundle.mkdir()
    deployment.manifest = {'commit': 'fixture'}
    deployment.fingerprints = {key: 'same' for key in ('api', 'clio', 'prophet', 'intelligence', 'configs', 'nginx', 'alloy')}
    deployment.previous_fingerprints = deployment.fingerprints.copy()
    deployment.migrations = []
    (deployment.bundle / 'fingerprints.tsv').write_text(''.join(f'{key} {value}\n' for key, value in deployment.fingerprints.items()))
    (deployment.bundle / 'release.json').write_text(json.dumps(deployment.manifest))
    (deployment.bundle / 'images.env').write_text('pinned-images')
    (deployment.bundle / 'docker-compose.yml').write_text('{}')
    (deployment.bundle / 'argus').write_bytes((ROOT / 'argus').read_bytes())
    (deployment.bundle / 'scripts').mkdir()
    (tmp_path / 'scripts').mkdir()
    for name in ('run', 'logs'):
        (deployment.bundle / 'scripts' / name).write_bytes((ROOT / 'scripts' / name).read_bytes())
    for name in ('configs', 'nginx', 'alloy'):
        (deployment.bundle / name).mkdir()
    deployment.snapshot = lambda: None
    deployment.run = lambda *args, **kwargs: deployment.calls.append(args)
    deployment.apply()
    assert (tmp_path / '.release.json').read_text() == json.dumps(deployment.manifest)
    assert (tmp_path / '.release-images.env').read_text() == 'pinned-images'
    assert (tmp_path / '.release-fingerprints.tsv').read_text() == (deployment.bundle / 'fingerprints.tsv').read_text()
    starts = [call for call in deployment.calls if call[0] == 'up']
    assert all('--no-deps' in call for call in starts)
    assert not any(service in call for call in starts for service in rolling.HTTP)
    assert sum('clio-green' in call for call in starts) == 1
    deployment.calls.clear()
    deployment.apply()
    assert not any(call[0] == 'smoke' for call in deployment.calls)
