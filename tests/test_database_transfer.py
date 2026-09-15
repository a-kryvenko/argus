"""Host transfer orchestration with fake Docker; never contacts real databases."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def transfer(tmp_path):
    root, bundle, tools = [tmp_path / name for name in ('host with spaces', 'release', 'tools')]
    for directory in (root, bundle, tools):
        directory.mkdir()
    for name in ('argus', 'deploy.sh', 'transfer.sh', 'install-tools.sh'):
        shutil.copy2(ROOT / 'scripts/deployment' / name, bundle / name)
    for name in ('configs', 'nginx', 'alloy'):
        (bundle / name).mkdir()
    for name in ('.env', '.env.local', '.release-images.env', 'docker-compose.yml'):
        (root / name).write_text('# old fixture\n')
    (root / '.release-fingerprints.tsv').write_text('api old\nclio old\nprophet old\nintelligence old\n')
    for name in ('images.env', 'docker-compose.yml', 'release.json'):
        (bundle / name).write_text('{}')
    (bundle / 'fingerprints.tsv').write_text('api new\nclio new\nprophet new\nintelligence new\n')
    log = tmp_path / 'calls'
    docker = tools / 'docker'
    docker.write_text(f'#!{sys.executable}\n' + r'''import json, os, sys
args = sys.argv[1:]
with open(os.environ['CALL_LOG'], 'a') as output:
    output.write(json.dumps(args) + '\n')
stage = os.getenv('FAIL_STAGE', '')
joined = ' '.join(args)
if 'transfer-databases.py' in joined:
    command = args[args.index('/var/www/scripts/transfer-databases.py') + 1]
    if command == stage:
        sys.exit(7)
    if command == 'preflight':
        print('source\tlegacy')
        for domain in ('api', 'clio', 'prophet', 'intelligence'):
            ready = '1' if domain in os.getenv('READY', '').split(',') else '0'
            print('\t'.join([domain, 'argus_' + domain, 'argus_' + domain, ready]))
elif args[-2:] == ['config', '--services']:
    print('api\nclio\nsolar-wind\ngeomagnetic\nclio-refresh\nclio-aggregate\nprophet\nprophet-api\nintelligence\npostgres\nredis')
elif 'pg_control_system()' in joined:
    print('123456789')
elif stage == 'restore' and 'pg_restore' in joined:
    sys.exit(7)
elif stage == 'backup' and 'pg_dumpall' in joined:
    sys.exit(7)
elif stage == 'deploy' and args[-4:] == ['run', '--rm', '--no-deps', 'clio-migrate']:
    sys.exit(7)
''')
    docker.chmod(0o755)
    rsync = tools / 'rsync'
    rsync.write_text('#!/bin/sh\nexit 0\n')
    rsync.chmod(0o755)
    env = {**os.environ, 'PATH': str(tools) + ':' + os.environ['PATH'], 'CALL_LOG': str(log)}
    subprocess.run(['bash', str(bundle / 'install-tools.sh'), str(root)], check=True, capture_output=True, env=env)
    def run(fail='', ready='', *arguments):
        return subprocess.run([str(root / 'bin/argus'), 'db', 'transfer', *arguments],
                              env={**env, 'FAIL_STAGE': fail, 'READY': ready}, capture_output=True, text=True)
    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return root, bundle, run, calls


def test_transfer_runs_all_phases_and_reuses_deployment_lock(transfer):
    root, _, run, calls = transfer
    result = run()
    assert result.returncode == 0, result.stderr
    lines = [' '.join(args) for args in calls()]
    index = lambda token: next(i for i, line in enumerate(lines) if token in line)
    assert index(' preflight') < index(' stop ') < index(' freeze') < index('pg_dumpall') < index(' provision')
    assert index(' snapshot') < index('pg_dump --') < index('pg_restore') < index(' verify')
    restore_indices = [i for i, line in enumerate(lines) if 'pg_restore' in line]
    verifies = [i for i, line in enumerate(lines) if 'transfer-databases.py verify' in line]
    migrate = index('run --rm --no-deps api-migrate')
    assert len(restore_indices) == len(verifies) == 4
    assert max(verifies) < migrate
    assert all(a < b for a, b in zip(restore_indices, verifies))
    assert all('--single-transaction' in lines[i] and '--no-owner' in lines[i] and '--no-privileges' in lines[i] for i in restore_indices)
    state = root / 'backups/database-transfer'
    assert (state / 'completed').exists() and (state / 'verified').exists()
    assert (state / 'cluster.sql').stat().st_mode & 0o777 == 0o600
    assert (state / 'original/.env').read_text() == '# old fixture\n'
    assert list((root / 'bin').iterdir()) == [root / 'bin/argus']


@pytest.mark.parametrize('stage', ['preflight', 'freeze', 'backup', 'provision', 'snapshot', 'restore', 'verify'])
def test_failures_do_not_apply_release_or_mark_transfer_verified(transfer, stage):
    root, _, run, calls = transfer
    result = run(fail=stage)
    assert result.returncode != 0
    assert not (root / 'backups/database-transfer/verified').exists()
    assert not any(args[-4:] == ['run', '--rm', '--no-deps', 'api-migrate'] for args in calls())
    if stage == 'preflight':
        assert not any('stop' in args for args in calls())


def test_retry_after_deployment_failure_never_recopies_source(transfer):
    root, _, run, calls = transfer
    assert run(fail='deploy').returncode != 0
    assert (root / 'backups/database-transfer/verified').exists()
    previous = len(calls())
    result = run()
    assert result.returncode == 0, result.stderr
    retry = [' '.join(args) for args in calls()[previous:]]
    assert any('transfer-databases.py identity' in line for line in retry)
    assert not any('pg_restore' in line or 'transfer-databases.py preflight' in line for line in retry)


def test_verified_existing_targets_are_not_restored_again(transfer):
    _, _, run, calls = transfer
    result = run(ready='api,clio')
    assert result.returncode == 0, result.stderr
    restores = [args for args in calls() if 'pg_restore' in ' '.join(args)]
    assert len(restores) == 2
    assert {args[-1] for args in restores} == {'argus_prophet', 'argus_intelligence'}


def test_changed_bundle_is_rejected_before_database_work(transfer):
    _, bundle, run, calls = transfer
    assert run(fail='preflight').returncode != 0
    previous = len(calls())
    (bundle / 'images.env').write_text('changed')
    assert run().returncode != 0
    assert len(calls()) == previous


def test_completed_transfer_is_a_noop_for_data_and_deployment(transfer):
    _, _, run, calls = transfer
    assert run().returncode == 0
    previous = len(calls())
    result = run()
    assert result.returncode == 0 and 'already completed' in result.stdout
    retry = [' '.join(args) for args in calls()[previous:]]
    assert not any('pg_restore' in line or 'run --rm --no-deps api-migrate' in line for line in retry)
