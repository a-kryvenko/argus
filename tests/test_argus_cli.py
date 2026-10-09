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
    folder = root / 'scripts'
    folder.mkdir(parents=True)
    (root / '.argus-mode').write_text('dev\n' if request.param == 'local' else 'prod\n')
    wrapper = folder / 'run'
    source = ROOT / 'scripts/run'
    wrapper.write_bytes(source.read_bytes())
    wrapper.chmod(0o755)
    for name in ('.env', '.env.local', '.release-images.env'):
        (root / name).write_text('# fixture\n')
    bindir = tmp_path / 'tools'
    bindir.mkdir()
    log = tmp_path / 'calls.jsonl'
    for name in ('docker',):
        executable = bindir / name
        executable.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
with open(os.environ['CALL_LOG'], 'a') as output:
    output.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), 'workdir': os.getenv('ARGUS_WORKDIR')}) + '\\n')
if 'ps' in sys.argv and os.getenv('POSTGRES_RUNNING'):
    print('existing-postgres-id')
if os.getenv('FAIL_POSTGRES') and 'up' in sys.argv:
    sys.exit(8)
if os.getenv('FAIL_CLIO') and any('clio' in arg for arg in sys.argv[1:]):
    sys.exit(7)
''')
        executable.chmod(0o755)
    environment = {**os.environ, 'PATH': str(bindir) + ':' + os.environ['PATH'], 'CALL_LOG': str(log)}
    def run(*args, fail=False, fail_postgres=False, running=False):
        result = subprocess.run([str(wrapper), *args], cwd=tmp_path, env={**environment, 'FAIL_CLIO': '1' if fail else '', 'FAIL_POSTGRES': '1' if fail_postgres else '', 'POSTGRES_RUNNING': '1' if running else ''}, capture_output=True, text=True)
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
    assert args[:1] == ['compose']
    assert args[args.index('--project-directory') + 1] == str(root)
    assert args[args.index('run'):args.index('run') + 4] == ['run', '--rm', '--no-deps', 'intelligence']
    assert (str(root / '.release-images.env') in args) == (mode == 'production')


@pytest.mark.parametrize('apply', [False, True])
def test_provision_defaults_to_plan(cli, apply):
    _, _, run = cli
    result, calls = run('db', 'provision', *(['--apply'] if apply else []))
    assert result.returncode == 0, result.stderr
    args = calls[0]['args']
    assert any(arg.endswith('/scripts/db/provision.py') for arg in args)
    assert ('--apply' in args) == apply


@pytest.mark.parametrize('fail', [False, True])
def test_all_migrations_are_sequential_and_stop_on_failure(cli, fail):
    mode, _, run = cli
    result, calls = run('db', 'migrate', fail=fail)
    assert result.returncode == (7 if fail else 0), result.stderr
    if mode == 'local':
        assert calls[0]['args'][-5:] == ['ps', '--status', 'running', '-q', 'postgres']
        assert calls[1]['args'][-6:] == ['up', '-d', '--wait', '--wait-timeout', '60', 'postgres']
        assert calls[-1]['args'][-2:] == ['stop', 'postgres']
        calls = calls[2:-1]
    assert len(calls) == (2 if fail else 4)
    for domain, call in zip(('api', 'clio', 'prophet', 'intelligence'), calls):
        args = call['args']
        assert args[-2:] == ['upgrade', 'head']
        target = domain if mode == 'local' else domain + '-migrate'
        assert args[args.index('--no-deps') + 1] == target


def test_specific_migration_uses_maintenance_service(cli):
    mode, _, run = cli
    result, calls = run('clio', 'migrate', 'current')
    assert result.returncode == 0, result.stderr
    if mode == 'local':
        assert calls[0]['args'][-5:] == ['ps', '--status', 'running', '-q', 'postgres']
        assert calls[1]['args'][-6:] == ['up', '-d', '--wait', '--wait-timeout', '60', 'postgres']
        assert calls[-1]['args'][-2:] == ['stop', 'postgres']
        calls = calls[2:-1]
    assert calls[0]['args'][-3:] == ['clio', 'migrate', 'current']
    if mode == 'production':
        assert calls[0]['args'][-4] == 'clio-migrate'


def test_unknown_commands_do_not_run_tools(cli):
    _, _, run = cli
    result, calls = run('db', 'migrate', '--apply')
    assert result.returncode == 2 and not calls


@pytest.mark.parametrize('cli', ['production'], indirect=True)
def test_production_maintenance_excludes_deployment_and_other_commands(cli):
    _, root, run = cli
    import fcntl
    with (root / '.deployment.lock').open('w') as held:
        fcntl.flock(held, fcntl.LOCK_SH | fcntl.LOCK_NB)
        result, calls = run('db', 'migrate')
        assert result.returncode != 0 and not calls


def test_compose_arguments_are_forwarded_without_running_app(cli):
    _, root, run = cli
    result, calls = run('compose', 'up', '-d', '--wait', 'postgres', 'redis')
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    assert calls[0]['args'][-6:] == [str(root / 'docker-compose.yml'), 'up', '-d', '--wait', 'postgres', 'redis']


@pytest.mark.parametrize('args', [('db', 'migrate'), ('prophet', 'generate', 'dst'), ('api', 'user', '--help')])
def test_deployment_lock_blocks_production_but_not_dev(cli, args):
    import fcntl
    mode, root, run = cli
    with (root / '.deployment.lock').open('w') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, calls = run(*args)
    if mode == 'production':
        assert result.returncode != 0 and not calls
    else:
        assert result.returncode == 0 and calls


@pytest.mark.parametrize('cli', ['local'], indirect=True)
@pytest.mark.parametrize('args', [('db', 'migrate'), ('clio', 'migrate', 'current')])
def test_database_start_failure_prevents_migrations(cli, args):
    _, _, run = cli
    result, calls = run(*args, fail_postgres=True)
    assert result.returncode == 8
    assert len(calls) == 3
    assert calls[-1]['args'][-2:] == ['stop', 'postgres']
    assert not any('run' in call['args'] for call in calls)


@pytest.mark.parametrize('cli', ['local'], indirect=True)
@pytest.mark.parametrize('fail', [False, True])
def test_existing_database_is_left_running(cli, fail):
    _, _, run = cli
    result, calls = run('db', 'migrate', running=True, fail=fail)
    assert result.returncode == (7 if fail else 0)
    assert calls[0]['args'][-5:] == ['ps', '--status', 'running', '-q', 'postgres']
    assert not any('up' in call['args'] or 'stop' in call['args'] for call in calls)
