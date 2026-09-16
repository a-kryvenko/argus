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
    backend = root / 'scripts' / request.param
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


@pytest.mark.parametrize('args', [('up',), ('ps',), ('api', 'refresh'), ('prophet', 'refresh', 'typo'), ('clio', 'refresh', 'solar-wind', '--watch'), ('clio', 'migrate', 'current')])
def test_invalid_commands_have_no_side_effects(entry, args):
    _, run = entry
    result, calls = run(*args)
    assert result.returncode == 2
    assert not calls


def test_clio_all_is_one_cycle_per_source(entry):
    _, run = entry
    result, calls = run('clio', 'refresh')
    assert result.returncode == 0
    assert calls == [['clio', 'collect', 'solar-wind'], ['clio', 'collect', 'geomagnetic'], ['clio', 'refresh']]


def test_refresh_stops_on_failure(entry):
    _, run = entry
    result, calls = run('clio', 'refresh', fail=True)
    assert result.returncode == 7
    assert len(calls) == 1


def test_shared_product_name(entry):
    _, run = entry
    result, calls = run('prophet', 'refresh', 'geomagnetic-activity')
    assert result.returncode == 0
    assert calls == [['prophet', 'generate', 'kp']]


def test_intelligence_refresh_all(entry):
    _, run = entry
    result, calls = run('intelligence', 'refresh')
    assert result.returncode == 0
    assert len(calls) == 6
    assert all(call[:2] == ['intelligence', 'process'] for call in calls)


def test_logs_routes_to_environment(entry):
    _, run = entry
    result, calls = run('logs', 'prophet', '--tail', '20', '-f')
    assert result.returncode == 0
    assert calls == [['prophet', '--tail', '20', '-f']]


@pytest.mark.parametrize('service', ['prophet', 'intelligence'])
def test_autogenerate_requires_service_metadata(entry, service):
    _, run = entry
    result, calls = run(service, 'migration', 'create', '-m', 'new fields', '--autogenerate')
    assert result.returncode == 2
    assert not calls
