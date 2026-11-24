"""Verify color selection, deployment ordering and failed cutovers."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/deployment'))
from _rollout import proxy
from _rollout.checksum import get_checksum
from _rollout.applications import Applications
from _rollout.compose import Compose
from _rollout.database import Database
from _rollout.proxy import NginxProxy
from _rollout.release import Release
from _rollout.applications import HTTP, WORKERS, slot_names
from _rollout.deployment import Deployment
from _rollout.state import DeploymentState, write_json


def harness(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    manifest = {'mode': 'rolling', 'checksum': get_checksum(bundle)}
    write_json(tmp_path / '.release_manifest', manifest)
    (tmp_path / '.active-color').write_text('blue\n')
    compose = Compose(tmp_path)
    applications = Applications(tmp_path, compose)
    deployment = Deployment(
        compose=compose,
        release=Release(tmp_path, bundle, compose.execute),
        database=Database(tmp_path, compose),
        applications=applications,
        proxy=NginxProxy(tmp_path, compose, applications),
        state=DeploymentState(tmp_path),
    )
    deployment.manifest = manifest
    deployment.color = 'blue'
    deployment.calls = []
    deployment.compose.run = lambda *args, **kwargs: deployment.calls.append(args) or ''
    deployment.compose.execute = lambda *args, **kwargs: deployment.calls.append(args)
    deployment.applications.check = lambda service, slot, routed=False: deployment.calls.append(('smoke', slot, routed))
    deployment.proxy.worker_pids = lambda service: set()
    deployment.applications.publish_static = lambda slot: None
    return deployment


def test_drain_waits_for_pre_reload_worker_even_before_signal_is_processed(tmp_path, monkeypatch):
    deployment = harness(tmp_path)
    generations = iter([{'100', '101'}, {'100', '102'}, {'102'}])
    deployment.proxy.worker_pids = lambda service: next(generations)
    sleeps = []
    monkeypatch.setattr(proxy.time, 'sleep', lambda duration: sleeps.append(duration))
    deployment.proxy.drain('clio', {'100'})
    assert len(sleeps) == 2


@pytest.mark.parametrize('version,keepalive', [('1.24.0', False), ('1.27.3', True)])
def test_existing_router_preserves_docker_dns_on_both_nginx_versions(tmp_path, version, keepalive):
    deployment = harness(tmp_path)

    def run(*args, **kwargs):
        if args[-1] == 'nginx -v 2>&1':
            return f'nginx version: nginx/{version}'
        return ''

    deployment.compose.run = run
    deployment.proxy.prepare('blue')
    deployment.proxy.switch('green')
    route = (tmp_path / 'routes/internal/clio.conf').read_text()
    assert 'X-Argus-Upstream clio-green' in route
    assert 'resolver 127.0.0.11 valid=5s' in route
    if keepalive:
        assert 'server clio-green:8000 resolve;' in route
        assert 'keepalive 16;' in route
    else:
        assert 'set $backend clio-green:8000;' in route
        assert 'proxy_pass http://$backend;' in route


def test_backup_retention_counts_old_sql_and_compressed_archives_together(tmp_path):
    deployment = harness(tmp_path)
    directory = tmp_path / 'backups'
    directory.mkdir()
    files = [directory / f'pre-migration-202610{day:02d}T000000.sql{suffix}'
             for day, suffix in [(1, ''), (2, '.gz'), (3, ''), (4, '.gz'), (5, '.gz')]]
    protected = [directory / 'manual.sql', directory / 'pre-migration-20261006T000000.sql.gz.partial']
    for path in files + protected:
        path.write_text('backup')
    (directory / 'pre-migration-20261007T000000.sql').mkdir()
    deployment.database.rotate_backups()
    assert all(not path.exists() for path in files[:2])
    assert all(path.exists() for path in files[2:] + protected)
    assert (directory / 'pre-migration-20261007T000000.sql').is_dir()


@pytest.mark.parametrize('failure', [False, True])
def test_backup_archive_is_verified_and_failed_dumps_remain_partial(tmp_path, failure):
    from backup import create_backup
    command = [sys.executable, '-c', 'import sys; print("database dump"); sys.exit(' + str(int(failure)) + ')']
    if failure:
        with pytest.raises(RuntimeError, match='Database dump failed'):
            create_backup(command, tmp_path)
        assert list(tmp_path.glob('*.partial'))
        assert not list(tmp_path.glob('*.sql.gz'))
    else:
        import gzip
        path = create_backup(command, tmp_path)
        assert gzip.decompress(path.read_bytes()) == b'database dump\n'
        assert path.stat().st_mode & 0o777 == 0o600
        assert not list(tmp_path.glob('*.partial'))


def test_checksum_and_installation_follow_current_bundle_layout(tmp_path):
    deployment = harness(tmp_path)
    for name in ('configs/project.yaml', 'data/models/model.bin', 'data/metrics/metrics.json', 'data/demo/current.json'):
        path = deployment.release.bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('release data')
    deployment.manifest = {'mode': 'rolling', 'release_id': 'a' * 32, 'checksum': get_checksum(deployment.release.bundle)}
    proxy_data = tmp_path / '.docker/nginx-proxy'
    proxy_data.mkdir(parents=True)
    (proxy_data / 'custom-proxy-config.conf').mkdir()
    (proxy_data / 'certificate.pem').write_text('operator certificate')
    bundled_proxy = deployment.release.bundle / '.docker/nginx-proxy'
    bundled_proxy.mkdir(parents=True)
    (bundled_proxy / 'custom-proxy-config.conf').write_text('release config')
    nginx = deployment.release.bundle / '.docker/nginx'
    nginx.mkdir()
    (nginx / 'default.conf').write_text('application config')
    deployment.manifest['checksum'] = get_checksum(deployment.release.bundle)
    write_json(tmp_path / '.release_manifest', deployment.manifest)
    (deployment.release.bundle / '__pycache__').mkdir()
    (deployment.release.bundle / '__pycache__/compiled.pyc').write_bytes(b'bytecode')
    deployment.release.verify(deployment.manifest)
    deployment.release.execute = lambda *args, **kwargs: subprocess.run(args, check=True, **kwargs)
    deployment.release.install()
    assert (tmp_path / 'data/models/model.bin').read_text() == 'release data'
    assert (proxy_data / 'custom-proxy-config.conf').is_dir()
    assert (proxy_data / 'certificate.pem').read_text() == 'operator certificate'
    assert (tmp_path / '.docker/nginx/default.conf').read_text() == 'application config'
    (deployment.release.bundle / 'data/models/model.bin').write_text('corrupt')
    with pytest.raises(RuntimeError, match='checksum'):
        deployment.release.verify(deployment.manifest)


def test_current_cron_layout_is_installed(tmp_path):
    deployment = harness(tmp_path)
    (tmp_path / 'cronjobs.txt').write_text('@reboot cd @ROOT@ && ./nginx-proxy-up.sh\n')
    legacy = tmp_path / 'apps/prophet/cronjobs.txt'
    legacy.parent.mkdir(parents=True)
    legacy.write_text('*/5 * * * * old-job\n')
    calls = []
    deployment.release.execute = lambda *args, **kwargs: calls.append((args, kwargs))
    deployment.release.install_crontab()
    assert calls == [(('crontab', '-'), {'input': f'@reboot cd {tmp_path} && ./nginx-proxy-up.sh\n', 'text': True})]


@pytest.mark.parametrize('failure', [None, 'startup', 'smoke', 'route', 'persist'])
def test_color_switch_commits_only_after_all_checks(tmp_path, monkeypatch, failure):
    deployment = harness(tmp_path)
    original = deployment.compose.run
    def compose(*args, **kwargs):
        original(*args, **kwargs)
        if failure == 'startup' and args[0] == 'up' and args[-1] == 'clio-green':
            raise RuntimeError('startup')
    deployment.compose.run = compose
    def smoke(service, slot, routed=False):
        deployment.calls.append(('smoke', slot, routed))
        if slot == 'api-green' and ((failure == 'smoke' and not routed) or (failure == 'route' and routed)):
            raise RuntimeError('smoke')
    deployment.applications.check = smoke
    if failure == 'persist':
        def fail_to_save(color):
            raise OSError('disk full')
        monkeypatch.setattr(deployment.state, 'save_color', fail_to_save)
    if failure:
        with pytest.raises((RuntimeError, OSError)):
            deployment.swap_blue_green_services()
        assert DeploymentState(tmp_path).read_color() == 'blue'
        assert ('stop', *slot_names('green')) in deployment.calls
        assert ('stop', *slot_names('blue')) not in deployment.calls
    else:
        deployment.swap_blue_green_services()
        assert DeploymentState(tmp_path).read_color() == 'green'
        assert deployment.calls.index(('smoke', 'frontend-green', False)) < deployment.calls.index(('smoke', 'clio-green', True))
        assert deployment.calls.index(('smoke', 'frontend-green', True)) < deployment.calls.index(('stop', *slot_names('blue')))


def test_failure_stopping_old_color_keeps_committed_color(tmp_path):
    deployment = harness(tmp_path)
    original = deployment.compose.run
    def compose(*args, **kwargs):
        original(*args, **kwargs)
        if args == ('stop', *slot_names('blue')):
            raise RuntimeError('stop failed')
    deployment.compose.run = compose
    with pytest.raises(RuntimeError, match='stop failed'):
        deployment.swap_blue_green_services()
    assert DeploymentState(tmp_path).read_color() == 'green'
    assert ('stop', *slot_names('green')) not in deployment.calls


def test_every_release_recreates_all_services_and_alternates_colors(tmp_path):
    deployment = harness(tmp_path)
    for expected_color in ('green', 'blue'):
        deployment.calls.clear()
        deployment.swap_blue_green_services()
        assert deployment.color == DeploymentState(tmp_path).read_color() == expected_color
        starts = [call for call in deployment.calls if call[0] == 'up']
        assert {call[-1] for call in starts} == set(slot_names(expected_color))
        assert all('--force-recreate' in call and '--pull' in call for call in starts)
        reloads = [call for call in deployment.calls if call[-2:] == ('-s', 'reload')]
        assert len(reloads) == 1


@pytest.mark.parametrize('mode', ['rolling', 'maintenance'])
def test_apply_selects_explicit_mode(tmp_path, monkeypatch, mode):
    calls = []
    deployment = Deployment(
        compose=SimpleNamespace(
            run=lambda *args: calls.append('pull'),
        ),
        release=SimpleNamespace(
            read_manifest=lambda: {'mode': mode},
            verify=lambda manifest: None,
            install=lambda: calls.append('install'),
        ),
        database=SimpleNamespace(),
        applications=SimpleNamespace(),
        proxy=SimpleNamespace(prepare=lambda color: calls.append('prepare')),
        state=SimpleNamespace(read_color=lambda **kwargs: 'blue', ensure_recovered=lambda: None),
    )
    monkeypatch.setattr(deployment, 'apply_rolling', lambda: calls.append('rolling'))
    monkeypatch.setattr(deployment, 'apply_maintenance', lambda: calls.append('maintenance'))
    deployment.apply()
    expected = ['install', 'pull']
    if mode == 'rolling':
        expected.append('prepare')
    assert calls == expected + [mode]


@pytest.mark.parametrize('fail', [False, True])
def test_rolling_never_backs_up_and_resumes_workers_after_failure(tmp_path, monkeypatch, fail):
    deployment = harness(tmp_path)
    deployment.compose.run = lambda *args, **kwargs: deployment.calls.append(args) or ('clio-worker' if args[0] == 'ps' else '')
    monkeypatch.setattr(deployment.database, 'backup', lambda: pytest.fail('Unexpected rolling backup'))
    monkeypatch.setattr(deployment, 'finish_release', lambda: None)
    def replace():
        if fail:
            raise RuntimeError('replacement failed')
    monkeypatch.setattr(deployment, 'swap_blue_green_services', replace)
    if fail:
        with pytest.raises(RuntimeError, match='replacement failed'):
            deployment.apply_rolling()
        assert deployment.calls[-1] == ('start', 'clio-worker')
    else:
        deployment.apply_rolling()
    migration = (str(tmp_path / 'argus'), 'db', 'migrate')
    assert [call for call in deployment.calls if call[0] == migration[0]] == [migration]
    assert deployment.calls.index(('stop', 'clio-worker')) < deployment.calls.index(migration)



@pytest.mark.parametrize('initial_color', [None, 'blue', 'green'])
@pytest.mark.parametrize('failure', [None, 'backup', 'migration'])
def test_maintenance_bootstraps_green_and_keeps_traffic_closed_after_migration_failure(tmp_path, monkeypatch, initial_color, failure):
    deployment = harness(tmp_path)
    (tmp_path / '.active-color').unlink()
    deployment.color = initial_color
    if initial_color:
        deployment.state.save_color(initial_color)
    deployment.manifest = {'mode': 'maintenance'}
    def backup():
        deployment.calls.append(('backup',))
        if failure == 'backup':
            raise RuntimeError('backup failed')
        return tmp_path / 'backup.sql.gz'
    def migrate():
        deployment.calls.append(('migrate',))
        if failure == 'migration':
            raise RuntimeError('migration failed')
    monkeypatch.setattr(deployment.database, 'backup', backup)
    monkeypatch.setattr(deployment.database, 'migrate', migrate)
    monkeypatch.setattr(deployment, 'finish_release', lambda: deployment.calls.append(('finish',)))
    monkeypatch.setattr(deployment.database, 'rotate_backups', lambda: deployment.calls.append(('cleanup',)))
    if failure:
        with pytest.raises(RuntimeError):
            deployment.apply_maintenance()
        assert (tmp_path / '.active-color').exists() == bool(initial_color)
        if initial_color:
            assert DeploymentState(tmp_path).read_color() == initial_color
    else:
        deployment.apply_maintenance()
        assert DeploymentState(tmp_path).read_color() == 'green'
        starts = [call for call in deployment.calls if call[0] == 'up' and '--force-recreate' in call]
        assert {call[-1] for call in starts} == set(slot_names('green'))
        assert deployment.calls.index(('migrate',)) < deployment.calls.index(('finish',)) < deployment.calls.index(('cleanup',))
    calls = deployment.calls
    maintenance_start = ('up', '-d', '--no-deps', '--pull', 'never',
                         '--scale', 'maintenance=1', '--wait', 'maintenance')
    stopped = ('stop', *WORKERS, *HTTP, *slot_names('blue'), *slot_names('green'))
    assert calls.index(maintenance_start) < calls.index(('stop', 'nginx')) < calls.index(stopped) < calls.index(('backup',))
    assert not any(call[0] == 'ps' for call in calls)
    assert 'postgres' not in stopped and 'redis' not in stopped
    needs_recovery = failure == 'migration' or (failure == 'backup' and initial_color is None)
    assert (tmp_path / '.maintenance-recovery.json').exists() == needs_recovery
    assert (('stop', 'maintenance') in calls) == (not needs_recovery)
    if failure is None:
        assert calls.index(('finish',)) < calls.index(('stop', 'maintenance')) < calls.index(('cleanup',))
    restores_previous = failure == 'backup' and initial_color is not None
    assert (('start', 'nginx') in calls) == restores_previous
    if restores_previous:
        assert ('start', *slot_names(initial_color), *WORKERS) in calls
        assert calls.index(('start', 'nginx')) < calls.index(('stop', 'maintenance'))


def test_maintenance_page_stays_up_when_final_startup_fails(tmp_path, monkeypatch):
    deployment = harness(tmp_path)
    monkeypatch.setattr(deployment.database, 'backup', lambda: tmp_path / 'backup.sql.gz')
    monkeypatch.setattr(deployment.database, 'migrate', lambda: None)
    def fail():
        raise RuntimeError('workers failed')
    monkeypatch.setattr(deployment, 'finish_release', fail)
    with pytest.raises(RuntimeError, match='Traffic remains closed'):
        deployment.apply_maintenance()
    assert (tmp_path / '.maintenance-recovery.json').exists()
    assert ('stop', 'maintenance') not in deployment.calls
    assert deployment.calls[-1] == ('stop', 'nginx', *WORKERS, *slot_names('green'))



def test_marker_is_required_for_rolling_but_not_for_maintenance(tmp_path, monkeypatch):
    deployment = harness(tmp_path)
    (tmp_path / '.active-color').unlink()
    def installed():
        raise RuntimeError('reached installation')
    monkeypatch.setattr(deployment.release, 'install', installed)
    for mode in ('rolling', 'maintenance'):
        manifest = {'mode': mode, 'checksum': get_checksum(deployment.release.bundle)}
        write_json(tmp_path / '.release_manifest', manifest)
        message = 'Missing .active-color' if mode == 'rolling' else 'reached installation'
        with pytest.raises(RuntimeError, match=message):
            deployment.apply()


def test_finish_release_keeps_existing_nginx_and_reloads_configuration(tmp_path, monkeypatch):
    deployment = harness(tmp_path)
    monkeypatch.setattr(deployment.release, 'install_crontab', lambda: deployment.calls.append(('crontab',)))
    deployment.finish_release()
    worker_start = deployment.calls[0]
    assert worker_start[0] == 'up' and '--wait' in worker_start
    assert set(worker_start[-len(WORKERS):]) == set(WORKERS)
    assert deployment.calls[1] == ('crontab',)
    nginx_up = [call for call in deployment.calls if call[0] == 'up' and 'nginx' in call]
    assert len(nginx_up) == 1
    assert '--no-recreate' in nginx_up[0] and '--pull' in nginx_up[0]
    assert not any(call[0] in ('stop', 'restart', 'rm') and 'nginx' in call for call in deployment.calls)
    assert ('exec', '-T', 'nginx', 'nginx', '-t') in deployment.calls
    assert ('exec', '-T', 'nginx', 'nginx', '-s', 'reload') in deployment.calls
