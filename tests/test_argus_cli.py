"""Unified CLI routing without starting services or connecting to databases."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=['local', 'production'])
def cli(tmp_path, request):
    root = tmp_path / 'project with spaces'
    folder = root / ('scripts' if request.param == 'local' else 'bin')
    folder.mkdir(parents=True)
    wrapper = folder / 'argus'
    source = ROOT / ('scripts/argus' if request.param == 'local' else 'scripts/deployment/argus')
    wrapper.write_bytes(source.read_bytes())
    wrapper.chmod(0o755)
    for domain in ('api', 'clio', 'prophet', 'intelligence'):
        executable = root / 'apps' / domain / '.venv/bin/python'
        executable.parent.mkdir(parents=True)
        executable.symlink_to(sys.executable)
    for name in ('.env', '.env.local', '.release-images.env'):
        (root / name).write_text('# fixture\n')
    bindir = tmp_path / 'tools'
    bindir.mkdir()
    log = tmp_path / 'calls.jsonl'
    for name in ('uv', 'docker'):
        executable = bindir / name
        executable.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
with open(os.environ['CALL_LOG'], 'a') as output:
    output.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), 'workdir': os.getenv('ARGUS_WORKDIR')}) + '\\n')
if os.getenv('FAIL_CLIO') and any('clio' in arg for arg in sys.argv[1:]):
    sys.exit(7)
''')
        executable.chmod(0o755)
    environment = {**os.environ, 'PATH': str(bindir) + ':' + os.environ['PATH'], 'CALL_LOG': str(log)}
    def run(*args, fail=False):
        result = subprocess.run([str(wrapper), *args], cwd=tmp_path, env={**environment, 'FAIL_CLIO': '1' if fail else ''}, capture_output=True, text=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls
    return request.param, root, run


def test_domain_arguments_and_environment_files(cli):
    mode, root, run = cli
    result, calls = run('intelligence', 'check', 'dst', '--release-id', 'value with spaces')
    assert result.returncode == 0, result.stderr
    args = calls[0]['args']
    assert args[-5:] == ['intelligence', 'check', 'dst', '--release-id', 'value with spaces']
    assert args.index(str(root / '.env')) < args.index(str(root / '.env.local'))
    if mode == 'local':
        assert '--no-sync' in args and '--frozen' in args
        assert args[args.index('--project') + 1] == str(root / 'apps/intelligence')
        assert calls[0]['cwd'] == str(root / 'apps/intelligence')
        assert calls[0]['workdir'] == str(root)
    else:
        assert str(root / '.release-images.env') in args


@pytest.mark.parametrize('apply', [False, True])
def test_provision_defaults_to_plan(cli, apply):
    _, _, run = cli
    result, calls = run('db', 'provision', *(['--apply'] if apply else []))
    assert result.returncode == 0, result.stderr
    args = calls[0]['args']
    assert any(arg.endswith('/scripts/provision-databases.py') for arg in args)
    assert ('--apply' in args) == apply


@pytest.mark.parametrize('fail', [False, True])
def test_all_migrations_are_sequential_and_stop_on_failure(cli, fail):
    mode, _, run = cli
    result, calls = run('db', 'migrate', fail=fail)
    assert result.returncode == (7 if fail else 0), result.stderr
    assert len(calls) == (2 if fail else 4)
    for domain, call in zip(('api', 'clio', 'prophet', 'intelligence'), calls):
        args = call['args']
        if mode == 'local':
            assert args[-2:] == ['upgrade', 'head']
            assert any(f'/apps/{domain}' in arg for arg in args)
        else:
            assert args[-1] == domain + '-migrate'


def test_specific_migration_uses_maintenance_service(cli):
    mode, _, run = cli
    result, calls = run('clio', 'migrate', 'current')
    assert result.returncode == 0, result.stderr
    assert calls[0]['args'][-3:] == ['clio', 'migrate', 'current']
    if mode == 'production':
        assert calls[0]['args'][-4] == 'clio-migrate'


def test_unknown_commands_do_not_run_tools(cli):
    _, _, run = cli
    result, calls = run('db', 'migrate', '--apply')
    assert result.returncode == 2 and not calls


def test_production_maintenance_excludes_deployment_and_other_commands(cli):
    mode, root, run = cli
    if mode != 'production':
        return
    import fcntl
    with (root / '.deployment.lock').open('w') as held:
        fcntl.flock(held, fcntl.LOCK_SH | fcntl.LOCK_NB)
        result, calls = run('db', 'migrate')
        assert result.returncode != 0 and not calls


def test_local_migrate_checks_all_environments_before_starting(cli):
    mode, root, run = cli
    if mode != 'local':
        return
    (root / 'apps/intelligence/.venv/bin/python').unlink()
    result, calls = run('db', 'migrate')
    assert result.returncode != 0 and not calls
    assert 'uv sync --project apps/intelligence --frozen' in result.stderr
