"""Migrations are scoped to api; runtime and migrations share one database owner."""
from alembic import context
from sqlalchemy import create_engine, pool, text
from app.db.base import Base
from app.db import models  # noqa: F401
from app.db.session import get_database_url

DOMAIN = 'api'
config = context.config


def include_name(name, type_, parent_names):
    if type_ == 'schema':
        return name == DOMAIN
    if type_ == 'table':
        return name != 'legacy_alembic_version'
    return True


def configure(**kwargs):
    context.configure(target_metadata=Base.metadata, compare_type=True,
                      include_schemas=True, include_name=include_name,
                      version_table_schema=DOMAIN, **kwargs)


if context.is_offline_mode():
    configure(url=get_database_url(), literal_binds=True,
              dialect_opts={'paramstyle': 'named'})
    with context.begin_transaction():
        context.execute('CREATE SCHEMA IF NOT EXISTS api')
        context.execute('SET search_path TO api, pg_catalog, pg_temp')
        context.run_migrations()
else:
    engine = create_engine(get_database_url(), poolclass=pool.NullPool,
                           connect_args={'options': '-csearch_path=api,pg_catalog,pg_temp'})
    with engine.connect() as connection:
        if connection.scalar(text("SELECT to_regclass('public.alembic_version')")):
            raise RuntimeError('Legacy public schema detected; restore the domain into its own database first')
        connection.execute(text('CREATE SCHEMA IF NOT EXISTS api'))
        connection.commit()
        # Reflect the domain explicitly, not twice as both the search-path
        # default and the named schema during autogeneration.
        connection.dialect.default_schema_name = 'public'
        configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()
