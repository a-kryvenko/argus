"""Dependency rules for the public source tree; no private checkout required."""
import ast
from pathlib import Path
import subprocess
import tomllib
import re

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_architecture_workflow_references_existing_local_paths():
    workflow = (ROOT / '.github/workflows/architecture.yml').read_text()
    paths = re.findall(r'(?<![\w/])(?:apps|packages|tests)/[\w/.-]+', workflow)
    assert paths
    # Additional checkouts are created by Actions, not present in a public clone.
    jobs = yaml.safe_load(workflow)['jobs']
    checkout_paths = {
        step.get('with', {}).get('path')
        for job in jobs.values() for step in job.get('steps', [])
        if step.get('uses', '').startswith('actions/checkout@')
    }
    missing = [path for path in paths if path not in checkout_paths and not (ROOT / path).exists()]
    assert not missing, f'Architecture workflow references missing paths: {missing}'


def imports(path):
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module


def test_package_dependency_direction():
    forbidden = {
        'common': {'app', 'clio', 'forecast', 'forecast_core', 'fastapi', 'sqlalchemy', 'psycopg', 'argus_prophet', 'argus_intelligence'},
        'forecast': {'app', 'clio', 'forecast_core', 'intelligence_core', 'fastapi', 'sqlalchemy', 'psycopg', 'argus_prophet', 'argus_intelligence'},
    }
    violations = []
    for package, blocked in forbidden.items():
        for path in (ROOT / 'packages' / package / 'src').rglob('*.py'):
            for module in imports(path):
                if module.split('.')[0] in blocked:
                    violations.append(f'{path.relative_to(ROOT)} imports {module}')
    assert not violations, '\n'.join(violations)


def dependency_names(config):
    dependencies = list(config['project']['dependencies'])
    for extra in config['project'].get('optional-dependencies', {}).values():
        dependencies.extend(extra)
    return {re.split(r'[\s\[<>=!~;@]', dependency, maxsplit=1)[0].lower().replace('_', '-')
            for dependency in dependencies}


@pytest.mark.parametrize(('path', 'allowed', 'required'), [
    ('packages/forecast', {'common'}, {'common'}),
    ('apps/api', {'common'}, {'common'}),
    ('apps/intelligence', {'common', 'intelligence-core'}, {'common', 'intelligence-core'}),
    ('apps/prophet', {'common', 'forecast', 'forecast-core'},
     {'common', 'forecast', 'forecast-core'}),
    ('packages/forecast-core', {'common', 'forecast'}, {'forecast'}),
])
def test_declared_package_boundaries(path, allowed, required):
    manifest = ROOT / path / 'pyproject.toml'
    if not manifest.exists() and path == 'packages/forecast-core':
        pytest.skip('Private checkout is not required for public CI')
    names = dependency_names(tomllib.loads(manifest.read_text()))
    internal = {'common', 'clio', 'forecast', 'forecast-core', 'intelligence-core',
                'argus-api', 'argus-clio', 'argus-prophet', 'argus-intelligence'}
    assert names & internal <= allowed, (path, names & internal - allowed)
    assert required <= names, (path, required - names)


def test_public_workspace_does_not_require_private_checkouts():
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    for member in config['tool']['uv']['workspace']['members']:
        assert (ROOT / member / 'pyproject.toml').is_file()
        assert 'core' not in member
    for package in ('common', 'forecast'):
        config = tomllib.loads((ROOT / 'packages' / package / 'pyproject.toml').read_text())
        for dependency in config['project']['dependencies']:
            assert not dependency.startswith(('forecast-core', 'intelligence-core'))


def test_private_source_is_not_in_public_tree():
    tracked = subprocess.check_output(
        ['git', 'ls-files', '--', 'packages/forecast-core', 'packages/intelligence-core'],
        cwd=ROOT, text=True,
    )
    assert not tracked
    result = subprocess.run(
        ['git', 'check-ignore', 'packages/forecast-core/src/example.py',
         'packages/intelligence-core/src/example.py'], cwd=ROOT,
        capture_output=True, text=True, check=True,
    )
    assert len(result.stdout.splitlines()) == 2


def test_runtime_sql_does_not_read_foreign_domain_tables():
    roots = {'api': ROOT / 'apps/api/app', 'clio': ROOT / 'apps/clio/src/clio',
             'prophet': ROOT / 'apps/prophet/src/argus_prophet',
             'intelligence': ROOT / 'apps/intelligence/src/argus_intelligence'}
    for owner, root in roots.items():
        for path in root.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for domain in re.findall(r'\b(?:FROM|JOIN|UPDATE|INTO|TABLE)\s+(api|clio|prophet|intelligence)\.', node.value, re.I):
                        assert domain.lower() == owner, (path, domain)


def test_api_environment_does_not_include_forecast_package():
    config = tomllib.loads((ROOT / 'apps/api/pyproject.toml').read_text())
    assert 'forecast' not in config['tool']['uv']['sources']
    assert '../../packages/forecast' not in config['tool']['uv']['workspace']['members']
    lock = tomllib.loads((ROOT / 'apps/api/uv.lock').read_text())
    assert not {'forecast', 'forecast-core', 'clio', 'argus-prophet', 'argus-clio', 'argus-intelligence'} & {package['name'] for package in lock['package']}


def test_clio_uses_base_backend_and_prophet_requests_models():
    clio = tomllib.loads((ROOT / 'apps/clio/pyproject.toml').read_text())
    prophet = tomllib.loads((ROOT / 'apps/prophet/pyproject.toml').read_text())
    assert any(d.startswith('forecast-core>=') for d in clio['project']['dependencies'])
    assert any(d.startswith('forecast-core[models]>=') for d in prophet['project']['dependencies'])


@pytest.mark.parametrize('root,blocked,private_surface', [
    ('apps/api/app', {'forecast', 'clio', 'forecast_core', 'argus_prophet', 'argus_intelligence'}, {'intelligence_core.api'}),
    ('apps/clio/src', {'app', 'argus_prophet', 'argus_intelligence', 'intelligence_core'}, {'forecast_core.calibration', 'forecast_core.observations'}),
    ('apps/prophet/src', {'app', 'clio', 'argus_intelligence', 'intelligence_core'}, {'forecast_core.api'}),
    ('apps/intelligence/src', {'app', 'clio', 'argus_prophet', 'forecast', 'forecast_core'}, {'intelligence_core.api'}),
    ('packages/forecast-core/src', {'clio', 'argus_prophet', 'argus_intelligence', 'app'}, None),
])
def test_service_import_boundaries(root, blocked, private_surface):
    for path in (ROOT / root).rglob('*.py'):
        for module in imports(path):
            assert module.split('.')[0] not in blocked, (path, module)
            if private_surface is not None and module.split('.')[0] in {'forecast_core', 'intelligence_core'}:
                assert module in private_surface, (path, module)


def test_prophet_environment_does_not_include_provider_library():
    lock = tomllib.loads((ROOT / 'apps/prophet/uv.lock').read_text())
    assert not {'clio', 'argus-clio'} & {package['name'] for package in lock['package']}


def test_private_inference_has_no_configuration_or_observation_file_access():
    root = ROOT / 'packages/forecast-core/src/forecast_core/inference'
    if not root.exists():
        pytest.skip('Private checkout unavailable')
    for path in root.rglob('*.py'):
        assert 'common.config' not in set(imports(path)), path
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert not (isinstance(node.func, ast.Name) and node.func.id == 'open'), path
