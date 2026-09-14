"""Deployment must not hand administrator or migration credentials to runtimes."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_domain_credentials_are_not_shared_with_runtime_containers():
    services = yaml.safe_load((ROOT / '.deploy/docker-compose.yml').read_text())['services']
    for name in ('api', 'clio', 'solar-wind', 'geomagnetic', 'clio-refresh', 'clio-aggregate', 'prophet', 'prophet-api'):
        service = services[name]
        assert 'env_file' not in service, name
        environment = service['environment']
        assert 'DB_USER' not in environment and 'DB_PASSWORD' not in environment, name
        assert not any('MIGRATION' in key for key in environment), name
        if name == 'api':
            assert 'API_DB_PASSWORD' in environment and 'CLIO_DB_PASSWORD' not in environment
        elif name in ('prophet', 'prophet-api'):
            assert 'PROPHET_DB_PASSWORD' in environment
            assert 'CLIO_DB_PASSWORD' not in environment and 'API_DB_PASSWORD' not in environment
        else:
            assert 'CLIO_DB_PASSWORD' in environment and 'API_DB_PASSWORD' not in environment
    for name in ('api-migrate', 'clio-migrate', 'prophet-migrate', 'db-bootstrap'):
        assert services[name]['profiles'] == ['maintenance']
    assert services['api']['environment']['OBSERVATIONS_URL'] == 'http://clio:8000'
    assert services['prophet']['environment']['OBSERVATIONS_URL'] == 'http://clio:8000'


def test_no_application_jobs_remain_in_cron():
    lines = [line for line in (ROOT / '.deploy/cronjobs.txt').read_text().splitlines()
             if line and not line.startswith('#')]
    assert len(lines) == 1
    assert lines[0].startswith('@reboot') and 'nginx-proxy-up.sh' in lines[0]


def test_forecast_read_service_is_internal_and_requires_token():
    services = yaml.safe_load((ROOT / '.deploy/docker-compose.yml').read_text())['services']
    reader = services['prophet-api']
    assert 'ports' not in reader
    assert set(reader['networks']) == {'backend', 'forecasts'}
    assert all('/data' not in volume for volume in reader['volumes'])
    assert reader['environment']['FORECASTS_SERVICE_TOKEN'].startswith('${FORECASTS_SERVICE_TOKEN:?')
    assert services['api']['environment']['FORECASTS_URL'] == 'http://prophet-api:8000'


def test_intelligence_credentials_and_networks_are_scoped():
    services = yaml.safe_load((ROOT / '.deploy/docker-compose.yml').read_text())['services']
    runtime = services['intelligence']
    assert 'profiles' not in runtime
    assert runtime['command'] == ['intelligence', 'worker']
    assert runtime['networks'] == ['backend', 'forecasts']
    assert set(runtime['environment']) == {'DB_HOST', 'DB_NAME', 'INTELLIGENCE_DB_PASSWORD', 'FORECASTS_URL', 'FORECASTS_SERVICE_TOKEN'}
    assert not any(key in runtime for key in ('volumes', 'ports', 'env_file'))
    assert services['intelligence-migrate']['profiles'] == ['maintenance']
    assert set(services['intelligence-migrate']['environment']) == {'DB_HOST', 'DB_NAME', 'INTELLIGENCE_MIGRATION_PASSWORD'}
    assert services['intelligence-provision']['profiles'] == ['maintenance']
    for name, service in services.items():
        if not name.startswith('intelligence'):
            assert not any(key.startswith('INTELLIGENCE_') for key in service.get('environment', {})), name
