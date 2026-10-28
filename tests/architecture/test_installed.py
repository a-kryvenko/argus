"""Installed service environments stay isolated from each other."""
from pathlib import Path
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('application,blocked', [
    ('api', {'clio', 'prophet', 'intelligence', 'forecast', 'forecast_core', 'intelligence_core'}),
    ('clio', {'api', 'prophet', 'intelligence', 'forecast', 'forecast_core', 'intelligence_core'}),
    ('prophet', {'app', 'api', 'clio', 'intelligence', 'intelligence_core'}),
    ('intelligence', {'api', 'clio', 'prophet', 'forecast', 'forecast_core'}),
])
def test_installed_application_imports_without_other_services(application, blocked, tmp_path):
    import os
    import subprocess
    code = f'''
import importlib
import importlib.util
for module in {sorted(blocked)!r}:
    assert importlib.util.find_spec(module) is None, module
importlib.import_module({application + '.main'!r})
'''
    if application == 'prophet':
        code += '''
import prophet.db.session
import prophet.services.generation.calculation
from forecast_core.api import AtmosphericDensityForecastService
'''
    result = subprocess.run([str(ROOT / 'apps' / application / '.venv/bin/python'), '-c', code],
                            cwd=tmp_path, env={**os.environ, 'PYTHONPATH': '', 'ARGUS_WORKDIR': str(ROOT)},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_project_test_environment_does_not_install_applications():
    import importlib.util
    config = tomllib.loads((ROOT / 'tests/runtime/pyproject.toml').read_text())
    locked = tomllib.loads((ROOT / 'tests/runtime/uv.lock').read_text())
    assert not {'api', 'clio', 'prophet', 'intelligence',
                'forecast-core', 'intelligence-core'} & {p['name'] for p in locked['package']}
    assert not config.get('tool', {}).get('uv', {}).get('sources')
    for module in ('api', 'clio', 'prophet', 'intelligence', 'forecast_core', 'intelligence_core'):
        assert importlib.util.find_spec(module) is None, module
