"""Static runtime dependency checks; no installed services required."""
from pathlib import Path
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_api_environment_does_not_include_forecast_package():
    config = tomllib.loads((ROOT / 'apps/api/pyproject.toml').read_text())
    assert 'forecast' not in config['tool']['uv']['sources']
    assert '../../packages/forecast' not in config['tool']['uv']['workspace']['members']


def test_clio_has_no_forecast_packages_and_prophet_requests_models():
    clio = tomllib.loads((ROOT / 'apps/clio/pyproject.toml').read_text())
    prophet = tomllib.loads((ROOT / 'apps/prophet/pyproject.toml').read_text())
    assert set(clio['tool']['uv']['sources']) == {'common'}
    assert any(d.startswith('forecast-core[models]>=') for d in prophet['project']['dependencies'])


@pytest.mark.parametrize('application,blocked', [
    ('clio', {'api', 'prophet', 'intelligence', 'forecast', 'forecast-core', 'intelligence-core'}),
    ('prophet', {'api', 'clio', 'intelligence', 'intelligence-core'}),
    ('intelligence', {'api', 'clio', 'prophet', 'forecast', 'forecast-core'}),
    ('api', {'clio', 'prophet', 'intelligence', 'forecast', 'forecast-core', 'intelligence-core'}),
])
def test_service_lockfiles_exclude_foreign_services_and_backends(application, blocked):
    lock = tomllib.loads((ROOT / 'apps' / application / 'uv.lock').read_text())
    assert not blocked & {package['name'] for package in lock['package']}
