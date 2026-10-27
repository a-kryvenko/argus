"""Migrations are scoped to api; runtime and migrations share one database owner."""

from common.db.alembic import run_migrations
from common.db.session import get_database_url

from api.db.base import Base
from api.db.models import dashboard, monitoring

run_migrations(
    'api',
    get_database_url('API'),
    Base.metadata
)