"""Public command contract is identical across installed environments."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=['dev', 'prod'])
def entry(tmp_path, request):
    root = tmp_path / 'installation with spaces'
    root.mkdir()
    shutil.copy2(ROOT / 'argus', root / 'argus')
    (root / '.argus-mode').write_text(request.param + '\n')
    backend = root / 'scripts'
    backend.mkdir(parents=True)
    log = root / 'calls'
    for command in ('run', 'logs'):
        p = backend / command
        p.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
with open(os.environ['ARGUS_TEST_CALLS'], 'a') as out:
    out.write(json.dumps(sys.argv[1:]) + '\\n')
if os.getenv('ARGUS_TEST_FAIL') and sys.argv[1:3] == ['clio', 'collect']:
    sys.exit(7)
''')
        p.chmod(0o755)
    def run(*args, fail=False):
        result = subprocess.run([str(root / 'argus'), *args], cwd=tmp_path,
            env={**os.environ, 'ARGUS_TEST_CALLS': str(log), 'ARGUS_TEST_FAIL': '1' if fail else ''},
            capture_output=True, text=True)
        return result, [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return request.param, run


@pytest.mark.parametrize('service', ['api', 'clio', 'prophet', 'intelligence'])
def test_service_migrate_applies_head(entry, service):
    _, run = entry
    result, calls = run(service, 'migrate')
    assert result.returncode == 0, result.stderr
    assert calls == [[service, 'migrate', 'upgrade', 'head']]


def test_create_migration_is_dev_only_and_preserves_message(entry):
    mode, run = entry
    result, calls = run('clio', 'migration', 'create', '-m', 'add run metadata', '--autogenerate')
    assert result.returncode == (0 if mode == 'dev' else 2)
    assert calls == ([['clio', 'migrate', 'revision', '-m', 'add run metadata', '--autogenerate']] if mode == 'dev' else [])


@pytest.mark.parametrize('args', [('up',), ('ps',), ('clio', 'migrate', 'current')])
def test_invalid_commands_have_no_side_effects(entry, args):
    _, run = entry
    result, calls = run(*args)
    assert result.returncode == 2
    assert not calls


def test_collect_stops_on_failure(entry):
    _, run = entry
    result, calls = run('clio', 'collect', fail=True)
    assert result.returncode == 7
    assert len(calls) == 1


def test_logs_routes_to_environment(entry):
    _, run = entry
    result, calls = run('logs', 'prophet', '--tail', '20', '-f')
    assert result.returncode == 0
    assert calls == [['prophet', '--tail', '20', '-f']]


@pytest.mark.parametrize('args', [(), ('january-2026',), ('--refresh-data',)])
def test_demo_is_one_command_in_both_environments(entry, args):
    _, run = entry
    result, calls = run('demo', *args)
    assert result.returncode == 0, result.stderr
    assert calls == [['prophet', 'demo', *args]]


@pytest.mark.parametrize('service', ['prophet', 'intelligence'])
def test_autogenerate_requires_service_metadata(entry, service):
    _, run = entry
    result, calls = run(service, 'migration', 'create', '-m', 'new fields', '--autogenerate')
    assert result.returncode == 2
    assert not calls


@pytest.mark.parametrize('args', [('observe',), ('observe', 'solar-wind-speed')])
def test_removed_observe_has_no_side_effects(entry, args):
    _, run = entry
    result, calls = run(*args)
    assert result.returncode == 2
    assert not calls


def test_compose_routes_to_selected_environment(entry):
    _, run = entry
    result, calls = run('compose', 'up', '-d', '--wait')
    assert result.returncode == 0, result.stderr
    assert calls == [['compose', 'up', '-d', '--wait']]


@pytest.mark.parametrize('args', [
    ('clio', 'collect'),
    ('prophet', 'verify', 'dst', '--days', '14'),
    ('intelligence', 'check', 'dst', '--release-id', 'value with spaces'),
    ('api', 'user', '--help'),
])
def test_service_arguments_pass_through_unchanged(entry, args):
    _, run = entry
    result, calls = run(*args)
    assert result.returncode == 0, result.stderr
    assert calls == [list(args)]
