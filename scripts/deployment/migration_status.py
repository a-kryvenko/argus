"""Read-only comparison of the database Alembic heads with this image."""
import importlib
from pathlib import Path
import sys

from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text


def revision_status(scripts, current, *, online=False):
    for revision in current:
        if scripts.get_revision(revision) is None:
            raise RuntimeError('Installed revision is absent from the candidate image')
    # Walk the complete upgrade path, including intermediate and merge revisions.
    revisions = list(scripts.iterate_revisions('heads', tuple(current), implicit_base=True))
    if online:
        blocked = sorted(revision.revision for revision in revisions
                         if getattr(revision.module, 'online_compatible', False) is not True)
        if blocked:
            return 'blocked:' + ','.join(blocked)
    return 'pending' if revisions else 'current'


def pending(domain, *, online=False):
    from common.db.session import get_database_url

    directory = Path(importlib.import_module(domain).__file__).parent / 'db' / 'migrations'
    scripts = ScriptDirectory(str(directory))
    engine = create_engine(get_database_url(domain.upper()),
                           connect_args={'connect_timeout': 5,
                                         'options': f'-csearch_path={domain},public,pg_catalog,pg_temp -cstatement_timeout=5000'})
    try:
        with engine.connect() as connection:
            if not inspect(connection).has_table('alembic_version'):
                current = set()
            else:
                current = set(connection.scalars(text('SELECT version_num FROM alembic_version')))
        return revision_status(scripts, current, online=online)
    finally:
        engine.dispose()


if __name__ == '__main__':
    try:
        print(pending(sys.argv[1], online='--online' in sys.argv[2:]), flush=True)
    except Exception:
        print('Could not compare database revisions with the candidate image', file=sys.stderr)
        sys.exit(1)
