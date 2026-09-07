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
