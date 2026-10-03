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

