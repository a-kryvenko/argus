"""Prophet migrations share the runtime database owner."""

from common.db.alembic import run_migrations
from common.db.session import get_database_url

run_migrations(
    'prophet',
    get_database_url('PROPHET'),
    None
)
