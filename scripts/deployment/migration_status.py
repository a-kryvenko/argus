"""Read-only comparison of the database Alembic heads with this image."""
import importlib
from pathlib import Path
import sys

from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text


def pending(domain):
    modules = {'api': ('app.db.session', 'get_database_url', None),
               'clio': ('clio.db.session', 'get_database_url', 'clio'),
               'prophet': ('argus_prophet.db.session', 'get_database_url', 'argus_prophet'),
               'intelligence': ('argus_intelligence.db', 'database_url', 'argus_intelligence')}
    module, getter, package = modules[domain]
    directory = Path('alembic') if package is None else Path(importlib.import_module(package).__file__).parent / 'migrations'
    scripts = ScriptDirectory(str(directory))
    expected = set(scripts.get_heads())
    engine = create_engine(getattr(importlib.import_module(module), getter)(),
                           connect_args={'connect_timeout': 5, 'options': '-cstatement_timeout=5000'})
    try:
        with engine.connect() as connection:
            if not inspect(connection).has_table('alembic_version', schema=domain):
                return True
            current = set(connection.scalars(text(f'SELECT version_num FROM {domain}.alembic_version')))
        for revision in current:
            if scripts.get_revision(revision) is None:
                raise RuntimeError('Installed revision is absent from the candidate image')
        return current != expected
    finally:
        engine.dispose()


if __name__ == '__main__':
    try:
        print('pending' if pending(sys.argv[1]) else 'current', flush=True)
    except Exception:
        print('Could not compare database revisions with the candidate image', file=sys.stderr)
        sys.exit(1)
