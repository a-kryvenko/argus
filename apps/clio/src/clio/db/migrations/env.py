"""Migrations are scoped to api; runtime and migrations share one database owner."""

from common.db.alembic import run_migrations
from common.db.session import get_database_url

from clio.db.base import Base
from clio.db.models import (
    file,
    file_observation,
    measurement,
    normalized_observation,
)

run_migrations(
    'clio',
    get_database_url('CLIO'),
    Base.metadata
)
