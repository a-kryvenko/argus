"""Migrations are scoped to intelligence; provision/adopt legacy storage first."""
from alembic import context
from sqlalchemy import create_engine, pool, text
from argus_intelligence.db import database_url as get_database_url

DOMAIN = 'intelligence'
config = context.config


def include_name(name, type_, parent_names):
    if type_ == 'schema':
        return name == DOMAIN
    if type_ == 'table':
        return name != 'legacy_alembic_version'
    return True


def configure(**kwargs):
    context.configure(target_metadata=None, compare_type=True,
                      include_schemas=True, include_name=include_name,
                      version_table_schema=DOMAIN, **kwargs)


if context.is_offline_mode():
    configure(url=get_database_url(migration=True), literal_binds=True,
              dialect_opts={'paramstyle': 'named'})
    with context.begin_transaction():
        context.execute('SET search_path TO intelligence, pg_catalog, pg_temp')
        context.run_migrations()
        context.execute('REVOKE ALL ON intelligence.alembic_version FROM argus_intelligence')
else:
    engine = create_engine(get_database_url(migration=True), poolclass=pool.NullPool,
                           connect_args={'options': '-csearch_path=intelligence,pg_catalog,pg_temp'})
    with engine.connect() as connection:
        if connection.scalar(text("SELECT to_regclass('public.alembic_version')")):
            raise RuntimeError('Run domain database bootstrap before domain migrations')
        connection.commit()
        # Reflect the domain explicitly, not twice as both the search-path
        # default and the named schema during autogeneration.
        connection.dialect.default_schema_name = 'public'
        configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()
            if connection.scalar(text("SELECT to_regclass('intelligence.alembic_version')")):
                connection.execute(text('REVOKE ALL ON intelligence.alembic_version FROM argus_intelligence'))
    engine.dispose()
