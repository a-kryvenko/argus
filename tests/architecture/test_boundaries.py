"""Dependency rules for the public source tree; no private checkout required."""
import ast
from pathlib import Path
import subprocess
import tomllib
import re

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


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
        'common': {'api', 'clio', 'forecast', 'forecast_core', 'intelligence_core', 'fastapi', 'sqlalchemy', 'psycopg', 'prophet', 'intelligence'},
        'forecast': {'api', 'clio', 'forecast_core', 'intelligence_core', 'fastapi', 'sqlalchemy', 'psycopg', 'prophet', 'intelligence'},
    }
    violations = []
    for package, blocked in forbidden.items():
        for path in (ROOT / 'packages' / package / 'src').rglob('*.py'):
            for module in imports(path):
                if module.split('.')[0] == 'sqlalchemy' and path.is_relative_to(
                    ROOT / 'packages/common/src/common/db'
                ):
                    continue
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
    internal = {'common', 'forecast', 'forecast-core', 'intelligence-core',
                'api', 'clio', 'prophet', 'intelligence'}
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


def sql_domains(source):
    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield from re.findall(r'\b(?:FROM|JOIN|UPDATE|INTO|TABLE)\s+(api|clio|prophet|intelligence)\.', node.value, re.I)


@pytest.mark.parametrize(('source', 'expected'), [
    ('"""Responses from Intelligence."""', []),
    ('def read():\n    """Responses from Intelligence."""\n    return "SELECT * FROM clio.measurement"', ['clio']),
    ('async def read():\n    """Responses from Intelligence."""\n    await execute("SELECT * FROM intelligence.profile")', ['intelligence']),
    ('class Client:\n    """Responses from Intelligence."""\n    query = "SELECT * FROM api.users JOIN prophet.release ON true"', ['api', 'prophet']),
])
def test_sql_domain_scan_ignores_only_docstrings(source, expected):
    assert list(sql_domains(source)) == expected


def test_runtime_sql_does_not_read_foreign_domain_tables():
    roots = {'api': ROOT / 'apps/api/src/api', 'clio': ROOT / 'apps/clio/src/clio',
             'prophet': ROOT / 'apps/prophet/src/prophet',
             'intelligence': ROOT / 'apps/intelligence/src/intelligence'}
    for owner, root in roots.items():
        for path in root.rglob('*.py'):
            if 'migrations' not in path.parts:
                assert not list(sql_domains(path.read_text())), (path, 'Runtime SQL must use search_path')
            for domain in sql_domains(path.read_text()):
                assert domain.lower() == owner, (path, domain)


@pytest.mark.parametrize('root,blocked,private_surface', [
    ('apps/api/src/api', {'forecast', 'clio', 'forecast_core', 'intelligence_core', 'prophet', 'intelligence'}, set()),
    ('apps/clio/src', {'api', 'prophet', 'intelligence', 'intelligence_core', 'forecast', 'forecast_core'}, set()),
    ('apps/prophet/src', {'api', 'clio', 'intelligence', 'intelligence_core'}, {'forecast_core'}),
    ('apps/intelligence/src', {'api', 'clio', 'prophet', 'forecast', 'forecast_core'}, {'intelligence_core.api'}),
    ('packages/forecast-core/src', {'intelligence_core', 'clio', 'prophet', 'intelligence', 'api'}, None),
    ('packages/intelligence-core/src', {'clio', 'prophet', 'intelligence', 'api', 'forecast_core'}, None),
])
def test_service_import_boundaries(root, blocked, private_surface):
    paths = list((ROOT / root).rglob('*.py'))
    if not paths and root.startswith('packages/'):
        pytest.skip('Private checkout unavailable')
    assert paths, root
    for path in paths:
        for module in imports(path):
            assert module.split('.')[0] not in blocked, (path, module)
            if private_surface is not None and module.split('.')[0] in {'forecast_core', 'intelligence_core'}:
                assert any(module == surface or module.startswith(surface + '.')
                           for surface in private_surface), (path, module)


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


def test_scripts_contain_only_infrastructure():
    for directory in ('training', 'revalidation', 'benchmarks', 'demo'):
        assert not (ROOT / 'scripts' / directory).exists()
    scientific = {'numpy', 'pandas', 'scipy', 'torch', 'matplotlib', 'mlflow'}
    for path in (ROOT / 'scripts').rglob('*.py'):
        assert not scientific & {module.split('.')[0] for module in imports(path)}, path


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
