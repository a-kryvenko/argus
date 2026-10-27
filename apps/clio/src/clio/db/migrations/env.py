"""Migrations are scoped to api; runtime and migrations share one database owner."""

from common.db.alembic import run_migrations
from common.db.session import get_database_url

from clio.db.base import Base
from clio.db.models import (
    file_observation,
    donki_poll,
    enlil_profile,
    goes_snapshot,
    gong_snapshot,
    measurement,
    normalized_observation,
    observation_source_status,
)

run_migrations(
    'clio',
    get_database_url('CLIO'),
    Base.metadata
)
