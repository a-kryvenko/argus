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
        elif isinstance(node, ast.Call) and node.args:
            dynamic = (isinstance(node.func, ast.Name) and node.func.id == '__import__') or (
                isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module')
            if dynamic and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                yield node.args[0].value


def test_package_dependency_direction():
    forbidden = {
        'common': {'app', 'clio', 'forecast', 'forecast_core', 'intelligence_core', 'fastapi', 'sqlalchemy', 'psycopg', 'argus_prophet', 'argus_intelligence'},
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
    ('apps/clio', {'common'}, {'common'}),
    ('apps/intelligence', {'common', 'intelligence-core'}, {'common', 'intelligence-core'}),
    ('packages/intelligence-core', {'common'}, {'common'}),
    ('apps/prophet', {'common', 'forecast', 'forecast-core'},
     {'common', 'forecast', 'forecast-core'}),
    ('packages/forecast-core', {'common', 'forecast'}, {'forecast'}),
])
def test_declared_package_boundaries(path, allowed, required):
    manifest = ROOT / path / 'pyproject.toml'
    if not manifest.exists() and path in {'packages/forecast-core', 'packages/intelligence-core'}:
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


def test_clio_has_no_forecast_packages_and_prophet_requests_models():
    clio = tomllib.loads((ROOT / 'apps/clio/pyproject.toml').read_text())
    prophet = tomllib.loads((ROOT / 'apps/prophet/pyproject.toml').read_text())
    assert not {'forecast', 'forecast-core'} & dependency_names(clio)
    assert set(clio['tool']['uv']['sources']) == {'common'}
    lock = tomllib.loads((ROOT / 'apps/clio/uv.lock').read_text())
    assert not {'forecast', 'forecast-core'} & {p['name'] for p in lock['package']}
    assert any(d.startswith('forecast-core[models]>=') for d in prophet['project']['dependencies'])


@pytest.mark.parametrize('root,blocked,private_surface', [
    ('apps/api/app', {'forecast', 'clio', 'forecast_core', 'intelligence_core', 'argus_prophet', 'argus_intelligence'}, set()),
    ('apps/clio/src', {'app', 'argus_prophet', 'argus_intelligence', 'intelligence_core', 'forecast', 'forecast_core'}, set()),
    ('apps/prophet/src', {'app', 'clio', 'argus_intelligence', 'intelligence_core'}, {'forecast_core.api'}),
    ('apps/intelligence/src', {'app', 'clio', 'argus_prophet', 'forecast', 'forecast_core'}, {'intelligence_core.api'}),
    ('packages/forecast-core/src', {'intelligence_core', 'clio', 'argus_prophet', 'argus_intelligence', 'app'}, None),
    ('packages/intelligence-core/src', {'clio', 'argus_prophet', 'argus_intelligence', 'app', 'forecast_core'}, None),
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


def test_research_namespaces_are_only_private_launchers():
    """Prevent proprietary recipes or tuning configs returning to the public tree."""
    for directory in ('scripts/revalidation', 'scripts/training/aia_wind',
                      'scripts/training/thermosphere'):
        for path in (ROOT / directory).iterdir():
            if not path.is_file():
                continue
            assert path.suffix == '.py', f'Private research asset in public tree: {path}'
            tree = ast.parse(path.read_text())
            assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                           for node in ast.walk(tree)), f'Implementation in research launcher: {path}'


def test_intelligence_runtime_is_infrastructure_only():
    """Public infrastructure may transport model output but cannot derive it."""
    derived_fields = {'estimated_altitude_loss_m', 'delta_v_loss_m_s', 'mean_density_kg_m3',
                      'mean_drag_accel_m_s2', 'mean_along_track_deceleration_m_s2',
                      'elevated_altitude_loss_m', 'high_altitude_loss_m'}
    for path in (ROOT / 'apps/intelligence/src').rglob('*.py'):
        assert not {'math', 'numpy', 'scipy'} & {module.split('.')[0] for module in imports(path)}, path
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            assert not (isinstance(node, ast.Attribute) and node.attr in derived_fields), (
                path, 'Derived impact fields belong to intelligence-core', node.attr)
    schema = ROOT / 'packages/common/src/common/schemas/leo_drag.py'
    for node in ast.walk(ast.parse(schema.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == 'consistent_series':
            assert not any(isinstance(child, ast.Attribute) and child.attr in {'thresholds', 'drag_risk'}
                           for child in ast.walk(node)), schema


def test_intelligence_audit_scripts_are_private_launchers():
    for name in ('drag_physics_audit.py', 'drag_density_transfer.py', 'drag_flight_audit.py'):
        path = ROOT / 'scripts/testing' / name
        tree = ast.parse(path.read_text())
        assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                       for node in ast.walk(tree)), path
        assert set(imports(path)) <= {'pathlib', 'runpy'}, path
    assert not (ROOT / 'apps/intelligence/tests/test_drag_physics_audit.py').exists()


@pytest.mark.parametrize('filename', ['docker-compose.yml', '.deploy/docker-compose.yml'])
def test_runtime_storage_credentials_and_startup_boundaries(filename):
    services = yaml.safe_load((ROOT / filename).read_text())['services']
    owners = {'clio': 'CLIO', 'clio-worker': 'CLIO', 'prophet': 'PROPHET',
              'prophet-api': 'PROPHET', 'intelligence': 'INTELLIGENCE',
              'intelligence-api': 'INTELLIGENCE', 'api': 'API'}
    allowed_data = {
        'clio': {'observations': False}, 'clio-worker': {'observations': False},
        'prophet': {'models': True, 'prophet': False, 'demo': False},
        'proswin': {'models': True, 'prophet': False},
        'prophet-api': {}, 'intelligence': {}, 'intelligence-api': {},
        'api': {'metrics': True, 'demo': True},
    }
    for name, allowed in allowed_data.items():
        service = services[name]
        actual = {}
        for mount in service.get('volumes', []):
            parts = mount.split(':')
            target = parts[1]
            assert target != '/var/www/data', (filename, name, mount)
            if target.startswith('/var/www/data/'):
                actual[target.removeprefix('/var/www/data/')] = len(parts) == 3 and parts[2] == 'ro'
        assert actual == allowed, (filename, name, actual)
        for dependency, settings in service.get('depends_on', {}).items():
            if dependency not in {'postgres', 'redis'}:
                assert settings['condition'] == 'service_started', (filename, name, dependency)
        if name in owners:
            environment = service['environment']
            db_keys = {key for key in environment if '_DB_' in key}
            assert db_keys and all(key.startswith(owners[name] + '_DB_') for key in db_keys)
            assert not {'DB_USER', 'DB_PASSWORD'} & environment.keys()
            assert service['depends_on']['postgres']['condition'] == 'service_healthy'
    for name in ('api', 'intelligence-api'):
        assert services[name]['environment']['INTELLIGENCE_SERVICE_TOKEN'] == (
            '${INTELLIGENCE_SERVICE_TOKEN:?INTELLIGENCE_SERVICE_TOKEN is required}')


@pytest.mark.parametrize('application,blocked', [
    ('clio', {'forecast', 'forecast-core', 'intelligence-core'}),
    ('prophet', {'intelligence-core', 'clio', 'argus-clio'}),
    ('intelligence', {'forecast', 'forecast-core', 'clio', 'argus-clio'}),
    ('api', {'forecast', 'forecast-core', 'intelligence-core', 'clio', 'argus-clio'}),
])
def test_service_lockfiles_exclude_foreign_backends(application, blocked):
    lock = tomllib.loads((ROOT / 'apps' / application / 'uv.lock').read_text())
    assert not blocked & {package['name'] for package in lock['package']}
