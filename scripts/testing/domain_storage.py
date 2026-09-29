"""The same isolated domain-storage checks for developers and CI."""
from contextlib import contextmanager
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


@contextmanager
def postgres():
    supplied = os.getenv('TEST_DATABASE_ADMIN_DSN')
    if supplied:
        yield supplied
        return
    name = 'argus-domain-tests-' + secrets.token_hex(6)
    password = secrets.token_hex(16)
    run('docker', 'run', '--rm', '-d', '--name', name,
        '--tmpfs', '/var/lib/postgresql/data', '-p', '127.0.0.1::5432',
        '-e', f'POSTGRES_PASSWORD={password}', 'postgres:17-alpine', stdout=subprocess.DEVNULL)
    try:
        port = run('docker', 'port', name, '5432/tcp', capture_output=True).stdout.strip().rsplit(':', 1)[1]
        for _ in range(60):
            ready = subprocess.run(['docker', 'exec', name, 'pg_isready', '-U', 'postgres'], capture_output=True)
            if ready.returncode == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError('Temporary PostgreSQL did not become ready within 60 seconds')
        yield f'postgresql://postgres:{password}@127.0.0.1:{port}/postgres'
    finally:
        run('docker', 'stop', name, stdout=subprocess.DEVNULL)


def main():
    os.chdir(ROOT)
    with tempfile.TemporaryDirectory(prefix='argus-domain-tests-') as directory, postgres() as dsn:
        workdir = Path(directory)
        shutil.copytree(ROOT / 'configs', workdir / 'configs')
        environment = {**os.environ, 'TEST_DATABASE_ADMIN_DSN': dsn,
                       'ARGUS_WORKDIR': directory, 'DEBUG': 'true',
                       'SENTRY_COLLECT_POINT': '', 'SENTRY_DSN': '',
                       'PYTHONPATH': '',
                       'MPLCONFIGDIR': str(workdir / 'matplotlib')}

        def python(*args):
            run(sys.executable, *args, env=environment)

        print('Preflight: service imports and migrations', flush=True)
        python('-c', """
import sys
from pathlib import Path
import tempfile
from sqlalchemy.ext.asyncio import create_async_engine
from clio.cli import main
from argus_prophet.cli import main
from argus_intelligence.cli import main
sys.path.insert(0, 'tests/integration')
from test_domain_storage import provisioned_database, migrate
with tempfile.TemporaryDirectory() as directory:
    with provisioned_database(Path(directory)) as (_, _, environment):
        migrate(environment)
""")
        print('Domain storage, ownership, scheduling and verification', flush=True)
        python('-m', 'pytest', '-q', 'tests/test_database_urls.py', 'tests/integration', *sys.argv[1:])
        for filename in ('test_solar_wind_aggregation.py', 'test_solar_wind_retention.py', 'test_observation_recovery.py'):
            python(str(ROOT / 'apps/clio/tests/integration' / filename))
        python('-m', 'pytest', '-q', 'apps/prophet/tests/test_observations.py', '-k', 'verification')
        python('apps/api/tests/integration/verify_dashboard.py')


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None
