"""Static checks for isolation between clio, prophet and intelligence."""
import ast
from pathlib import Path
import re
import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

import pytest

ROOT = Path(__file__).resolve().parents[1]
APPLICATIONS = {
    'clio': ('clio', 'clio'),
    'prophet': ('argus_prophet', 'argus-prophet'),
    'intelligence': ('argus_intelligence', 'argus-intelligence'),
}


@pytest.mark.parametrize('application', APPLICATIONS)
def test_application_does_not_import_other_applications(application):
    blocked_modules = {
        module
        for name, (module, _) in APPLICATIONS.items()
        if name != application
    }

    root = ROOT / 'apps' / application / 'src'
    paths = list(root.rglob('*.py'))

    assert paths, f'No application sources found in {root}'

    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]

            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported_modules = [node.module]

            else:
                continue

            for imported_module in imported_modules:
                top_level_module = imported_module.partition('.')[0]

                assert top_level_module not in blocked_modules, (
                    f'{path} imports application module {imported_module!r}'
                )


@pytest.mark.parametrize('application', APPLICATIONS)
def test_application_dependencies_exclude_other_applications(application):
    blocked = {
        canonicalize_name(package)
        for name, (_, package) in APPLICATIONS.items()
        if name != application
    }

    root = ROOT / 'apps' / application

    manifest = tomllib.loads(
        (root / 'pyproject.toml').read_text(encoding='utf-8')
    )

    dependencies = list(manifest['project']['dependencies'])

    for extra in manifest['project'].get('optional-dependencies', {}).values():
        dependencies.extend(extra)

    dependency_names = {
        canonicalize_name(Requirement(dependency).name)
        for dependency in dependencies
    }

    assert not dependency_names & blocked, (
        application,
        dependency_names & blocked,
    )

    lock = tomllib.loads(
        (root / 'uv.lock').read_text(encoding='utf-8')
    )

    locked_names = {
        canonicalize_name(package['name'])
        for package in lock['package']
    }

    assert not locked_names & blocked, (
        application,
        locked_names & blocked,
    )



@pytest.mark.parametrize('application,module', [
    ('api', 'app'), ('clio', 'clio'), ('prophet', 'argus_prophet'),
    ('intelligence', 'argus_intelligence'),
])
def test_installed_application_imports_without_other_services(application, module, tmp_path):
    import os
    import subprocess
    modules = {'app', 'clio', 'argus_prophet', 'argus_intelligence'}
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
    assert not {'argus-api', 'argus-clio', 'argus-prophet', 'argus-intelligence',
                'forecast-core', 'intelligence-core'} & {p['name'] for p in locked['package']}
    assert not config.get('tool', {}).get('uv', {}).get('sources')
    for module in ('app', 'clio', 'argus_prophet', 'argus_intelligence', 'forecast_core', 'intelligence_core'):
        assert importlib.util.find_spec(module) is None, module
