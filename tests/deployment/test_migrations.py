"""Validate compatibility over real Alembic upgrade graphs without a database."""
import importlib.util
from pathlib import Path

from alembic.script import ScriptDirectory
from alembic.util import CommandError
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('migration_status', ROOT / 'scripts/deployment/migration_status.py')
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)


def graph(tmp_path, revisions):
    versions = tmp_path / 'versions'
    versions.mkdir()
    for name, parent, declaration in revisions:
        (versions / f'{name}.py').write_text(
            f'revision = {name!r}\ndown_revision = {parent!r}\n{declaration}\n')
    return ScriptDirectory(str(tmp_path))


@pytest.mark.parametrize('declaration', ['', 'online_compatible = False', 'online_compatible = "true"'])
def test_unsafe_intermediate_revision_blocks_even_when_head_is_compatible(tmp_path, declaration):
    scripts = graph(tmp_path, [('initial', None, ''), ('middle', 'initial', declaration),
                               ('tip', 'middle', 'online_compatible = True')])
    assert status.revision_status(scripts, {'initial'}, online=True) == 'blocked:middle'
    assert status.revision_status(scripts, {'initial'}) == 'pending'


def test_applied_revisions_need_no_declaration_and_retries_are_current(tmp_path):
    scripts = graph(tmp_path, [('initial', None, ''), ('tip', 'initial', 'online_compatible = True')])
    assert status.revision_status(scripts, {'initial'}, online=True) == 'pending'
    assert status.revision_status(scripts, {'tip'}, online=True) == 'current'
    assert status.revision_status(scripts, set(), online=True) == 'blocked:initial'


def test_merge_checks_only_unapplied_branches_and_merge_revision(tmp_path):
    scripts = graph(tmp_path, [('initial', None, ''), ('left', 'initial', ''),
                               ('right', 'initial', 'online_compatible = True'),
                               ('merge', ('left', 'right'), '')])
    assert status.revision_status(scripts, {'left'}, online=True) == 'blocked:merge'
    assert status.revision_status(scripts, {'left', 'right'}, online=True) == 'blocked:merge'


def test_unknown_database_revision_is_rejected(tmp_path):
    scripts = graph(tmp_path, [('initial', None, 'online_compatible = True')])
    with pytest.raises(CommandError, match='unknown'):
        status.revision_status(scripts, {'unknown'}, online=True)
