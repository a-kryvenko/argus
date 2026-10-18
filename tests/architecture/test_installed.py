"""Installed service environments stay isolated from each other."""
from pathlib import Path
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('application,module', [
    ('api', 'api'), ('clio', 'clio'), ('prophet', 'prophet'),
    ('intelligence', 'intelligence'),
])
def test_installed_application_imports_without_other_services(application, module, tmp_path):
    import os
    import subprocess
    modules = {'api', 'clio', 'prophet', 'intelligence'}
    blocked = modules - {module}
    code = f'''
import importlib
import importlib.util
for module in {sorted(blocked)!r}:
    assert importlib.util.find_spec(module) is None, module
importlib.import_module({module + '.main'!r})
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
