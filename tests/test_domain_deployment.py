"""Deployment must not hand administrator or migration credentials to runtimes."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_domain_credentials_are_scoped_and_migrations_use_the_same_owner():
    services = yaml.safe_load((ROOT / '.deploy/docker-compose.yml').read_text())['services']
    domains = {'api': ['api'], 'clio': ['clio', 'clio-worker'],
               'prophet': ['prophet', 'prophet-api'], 'intelligence': ['intelligence']}
    for domain, names in domains.items():
        for name in [*names, domain + '-migrate']:
            service = services[name]
            assert 'env_file' not in service
            environment = service['environment']
            database_keys = {key for key in environment if '_DB_' in key or key.startswith('DB_') or 'DATABASE' in key}
            expected = {domain.upper() + '_DB_' + field for field in ('HOST', 'PORT', 'NAME', 'USER', 'PASSWORD')}
            assert database_keys == expected, name
            for key in expected:
                assert environment[key].startswith('${' + key + ':'), name
            assert all(
                environment[key] == services[domain + '-migrate']['environment'][key] for key in expected)
        assert services[domain + '-migrate']['profiles'] == ['maintenance']
    assert services['db-provision']['profiles'] == ['maintenance']
    assert 'db-bootstrap' not in services and 'intelligence-provision' not in services
    assert services['api']['environment']['OBSERVATIONS_URL'] == 'http://clio:8000'
    assert services['prophet']['environment']['OBSERVATIONS_URL'] == 'http://clio:8000'
    assert services['clio-worker']['command'] == ['clio', 'worker']
    assert services['clio-worker']['healthcheck']['test'] == ['CMD', 'clio', 'check-health', 'worker']
    assert not {'solar-wind', 'geomagnetic', 'clio-refresh', 'clio-aggregate'} & services.keys()


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
    assert set(runtime['environment']) == {'INTELLIGENCE_DB_' + field for field in ('HOST', 'PORT', 'NAME', 'USER', 'PASSWORD')} | {'FORECASTS_URL', 'FORECASTS_SERVICE_TOKEN'}
    assert not any(key in runtime for key in ('volumes', 'ports', 'env_file'))
    assert services['intelligence-migrate']['profiles'] == ['maintenance']
    assert set(services['intelligence-migrate']['environment']) == {'INTELLIGENCE_DB_' + field for field in ('HOST', 'PORT', 'NAME', 'USER', 'PASSWORD')}
    for name, service in services.items():
        if not name.startswith('intelligence') and name != 'db-provision':
            assert not any(key.startswith('INTELLIGENCE_') for key in service.get('environment', {})), name
