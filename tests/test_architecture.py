"""Dependency rules for the public source tree; no private checkout required."""
import ast
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def imports(path):
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module


def test_package_dependency_direction():
    forbidden = {
        'common': {'app', 'clio', 'forecast', 'forecast_core', 'intelligence_core', 'fastapi', 'sqlalchemy'},
        'clio': {'app', 'forecast', 'forecast_core', 'intelligence_core', 'fastapi', 'sqlalchemy'},
        'forecast': {'app', 'clio', 'intelligence_core', 'fastapi', 'sqlalchemy'},
    }
    violations = []
    for package, blocked in forbidden.items():
        for path in (ROOT / 'packages' / package / 'src').rglob('*.py'):
            for module in imports(path):
                if module.split('.')[0] in blocked:
                    violations.append(f'{path.relative_to(ROOT)} imports {module}')
    assert not violations, '\n'.join(violations)


def test_api_only_uses_private_integration_surface():
    paths = [*(ROOT / 'apps/api/app').rglob('*.py'),
             *(ROOT / 'packages/forecast/src').rglob('*.py')]
    for path in paths:
        for module in imports(path):
            if module.startswith(('forecast_core', 'intelligence_core')):
                assert module in {'forecast_core.api', 'intelligence_core.api'}, (path, module)


def test_public_workspace_does_not_require_private_checkouts():
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    for member in config['tool']['uv']['workspace']['members']:
        assert (ROOT / member / 'pyproject.toml').is_file()
        assert 'core' not in member
    for package in ('common', 'clio', 'forecast'):
        config = tomllib.loads((ROOT / 'packages' / package / 'pyproject.toml').read_text())
        for dependency in config['project']['dependencies']:
            assert not dependency.startswith(('forecast-core', 'intelligence-core'))


def test_private_source_is_not_in_public_tree():
    assert not (ROOT / 'packages/intelligence-core').exists()
    result = subprocess.run(
        ['git', 'check-ignore', 'packages/forecast-core/src/example.py',
         'private/intelligence-core/src/example.py'], cwd=ROOT,
        capture_output=True, text=True, check=True,
    )
    assert len(result.stdout.splitlines()) == 2


def test_prophet_has_no_storage_or_api_imports():
    blocked = {'app', 'sqlalchemy', 'psycopg', 'alembic', 'clio'}
    for path in (ROOT / 'apps/prophet/src').rglob('*.py'):
        for module in imports(path):
            assert module.split('.')[0] not in blocked, (path, module)
            if module.startswith(('forecast_core', 'intelligence_core')):
                assert module == 'forecast_core.api', (path, module)
    config = tomllib.loads((ROOT / 'apps/prophet/pyproject.toml').read_text())
    for dependency in config['project']['dependencies']:
        assert not dependency.startswith(('argus-api', 'sqlalchemy', 'psycopg', 'alembic'))


def test_api_no_longer_owns_forecast_commands():
    assert not list((ROOT / 'apps/api/app/commands').glob('generate*forecast.py'))
    for path in (ROOT / 'apps/api/app').rglob('*.py'):
        for module in imports(path):
            assert not module.startswith('argus_prophet'), (path, module)


def test_api_does_not_import_observation_storage_or_private_backend():
    for path in (ROOT / 'apps/api/app').rglob('*.py'):
        for module in imports(path):
            assert module.split('.')[0] not in {'clio', 'argus_clio', 'forecast_core', 'argus_prophet'}, (path, module)
    assert {p.stem for p in (ROOT / 'apps/api/app/db/models').glob('*.py')} == {'__init__', 'dashboard'}
    assert not list((ROOT / 'apps/api/app/commands').glob('collect*.py'))


def test_clio_owns_storage_and_uses_only_private_adapter():
    for path in (ROOT / 'apps/clio/src/argus_clio').rglob('*.py'):
        for module in imports(path):
            assert module.split('.')[0] not in {'app', 'argus_prophet', 'intelligence_core'}, (path, module)
            if module.startswith('forecast_core'):
                assert path.name == 'calibration.py' and module == 'forecast_core.api', (path, module)
