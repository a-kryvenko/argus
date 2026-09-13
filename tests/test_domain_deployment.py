"""Deployment must not hand administrator or migration credentials to runtimes."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_domain_credentials_are_not_shared_with_runtime_containers():
    services = yaml.safe_load((ROOT / '.deploy/docker-compose.yml').read_text())['services']
    for name in ('api', 'clio', 'solar-wind', 'geomagnetic', 'clio-refresh', 'clio-aggregate', 'prophet'):
        service = services[name]
        assert 'env_file' not in service, name
        environment = service['environment']
        assert 'DB_USER' not in environment and 'DB_PASSWORD' not in environment, name
        assert not any('MIGRATION' in key for key in environment), name
        if name == 'api':
            assert 'API_DB_PASSWORD' in environment and 'CLIO_DB_PASSWORD' not in environment
        elif name == 'prophet':
            assert not any(key.endswith('DB_PASSWORD') for key in environment)
        else:
            assert 'CLIO_DB_PASSWORD' in environment and 'API_DB_PASSWORD' not in environment
    for name in ('api-migrate', 'clio-migrate', 'db-bootstrap'):
        assert services[name]['profiles'] == ['maintenance']
    assert services['api']['environment']['OBSERVATIONS_URL'] == 'http://clio:8000'
    assert services['prophet']['environment']['OBSERVATIONS_URL'] == 'http://clio:8000'


def test_no_application_jobs_remain_in_cron():
    lines = [line for line in (ROOT / '.deploy/cronjobs.txt').read_text().splitlines()
             if line and not line.startswith('#')]
    assert len(lines) == 1
    assert lines[0].startswith('@reboot') and 'nginx-proxy-up.sh' in lines[0]
