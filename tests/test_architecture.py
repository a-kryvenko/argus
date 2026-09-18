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
        'common': {'app', 'clio', 'forecast', 'forecast_core', 'fastapi', 'sqlalchemy', 'psycopg', 'argus_clio', 'argus_prophet'},
        'clio': {'app', 'forecast', 'forecast_core', 'fastapi', 'sqlalchemy', 'psycopg', 'argus_clio', 'argus_prophet'},
        'forecast': {'app', 'clio', 'intelligence_core', 'fastapi', 'sqlalchemy', 'psycopg', 'argus_clio', 'argus_prophet'},
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


def test_prophet_only_accesses_its_own_storage():
    blocked = {'app', 'argus_clio', 'clio'}
    for path in (ROOT / 'apps/prophet/src').rglob('*.py'):
        for module in imports(path):
            assert module.split('.')[0] not in blocked, (path, module)
            if module.startswith(('forecast_core', 'intelligence_core')):
                assert module == 'forecast_core.api', (path, module)
    config = tomllib.loads((ROOT / 'apps/prophet/pyproject.toml').read_text())
    for dependency in config['project']['dependencies']:
        assert not dependency.startswith(('argus-api', 'argus-clio'))


def test_api_no_longer_owns_forecast_commands():
    assert not list((ROOT / 'apps/api/app/commands').glob('generate*forecast.py'))
    for path in (ROOT / 'apps/api/app').rglob('*.py'):
        for module in imports(path):
            assert not module.startswith('argus_prophet'), (path, module)


def test_api_does_not_import_observation_storage_or_private_backend():
    for path in (ROOT / 'apps/api/app').rglob('*.py'):
        for module in imports(path):
            assert module.split('.')[0] not in {'forecast', 'clio', 'argus_clio', 'forecast_core', 'argus_prophet'}, (path, module)
    # API owns dashboard identities, monitoring snapshots and traffic aggregates;
    # observation and forecast storage remain in their respective services.
    assert {p.stem for p in (ROOT / 'apps/api/app/db/models').glob('*.py')} == {
        '__init__', 'dashboard', 'monitoring',
    }
    assert not list((ROOT / 'apps/api/app/commands').glob('collect*.py'))


def test_clio_owns_storage_and_uses_only_private_adapter():
    for path in (ROOT / 'apps/clio/src/argus_clio').rglob('*.py'):
        for module in imports(path):
            assert module.split('.')[0] not in {'app', 'argus_prophet', 'intelligence_core'}, (path, module)
            if module.startswith('forecast_core'):
                assert path.name == 'calibration.py' and module == 'forecast_core.api', (path, module)


def test_runtime_sql_does_not_read_foreign_domain_tables():
    import re
    roots = {'api': ROOT / 'apps/api/app', 'clio': ROOT / 'apps/clio/src/argus_clio',
             'prophet': ROOT / 'apps/prophet/src/argus_prophet',
             'intelligence': ROOT / 'apps/intelligence/src/argus_intelligence'}
    for owner, root in roots.items():
        for path in root.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for domain in re.findall(r'\b(?:FROM|JOIN|UPDATE|INTO|TABLE)\s+(api|clio|prophet|intelligence)\.', node.value, re.I):
                        assert domain.lower() == owner, (path, domain)


def test_intelligence_uses_shared_contracts_and_owns_its_storage():
    for path in (ROOT / 'apps/intelligence/src').rglob('*.py'):
        for module in imports(path):
            assert module.split('.')[0] not in {
                'app', 'argus_clio', 'argus_prophet', 'clio', 'forecast',
                'forecast_core',
            }, (path, module)
    config = tomllib.loads((ROOT / 'apps/intelligence/pyproject.toml').read_text())
    assert set(config['project']['dependencies']) == {'common', 'intelligence-core>=0.1.0', 'httpx>=0.28,<1', 'psycopg[binary]>=3.2,<4', 'sqlalchemy>=2.0,<3', 'alembic>=1.16,<2'}


def test_existing_domains_do_not_depend_on_intelligence_runtime():
    for root in ('apps/api/app', 'apps/clio/src', 'apps/prophet/src', 'packages/common/src'):
        for path in (ROOT / root).rglob('*.py'):
            assert all(module.split('.')[0] != 'argus_intelligence' for module in imports(path)), path


def test_api_environment_does_not_include_forecast_package():
    config = tomllib.loads((ROOT / 'apps/api/pyproject.toml').read_text())
    assert 'forecast' not in config['project']['dependencies']
    assert 'forecast' not in config['tool']['uv']['sources']
    assert '../../packages/forecast' not in config['tool']['uv']['workspace']['members']
    lock = tomllib.loads((ROOT / 'apps/api/uv.lock').read_text())
    assert not {'forecast', 'forecast-core', 'clio'} & {package['name'] for package in lock['package']}
