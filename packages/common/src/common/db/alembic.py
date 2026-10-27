from alembic import context
from sqlalchemy import create_engine, pool, text


def _configure(schema: str, metadata, **kwargs):
    def _include_name(name, type_, parent_names):
        if type_ == 'schema':
            return name == schema
        if type_ == 'table':
            return name != 'legacy_alembic_version'
        
        return True

    context.configure(
        target_metadata=metadata,
        compare_type=True,
        include_schemas=True,
        include_name=_include_name,
        version_table_schema=schema,
        **kwargs
    )

def run_migrations(schema: str, url, metadata):
    if context.is_offline_mode():
        _configure(
            schema=schema,
            metadata=metadata,
            url=url,
            literal_binds=True,
            dialect_opts={'paramstyle': 'named'}
        )
        with context.begin_transaction():
            context.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            context.execute(f"SET search_path TO {schema}, pg_catalog, pg_temp")
            context.run_migrations()
    else:
        engine = create_engine(
            url=url,
            poolclass=pool.NullPool,
            connect_args={'options': f"-csearch_path={schema},pg_catalog,pg_temp -clock_timeout=5000 -cstatement_timeout=300000"}
        )
        with engine.connect() as connection:
            connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            connection.commit()
            # Reflect the domain explicitly, not twice as both the search-path
            # default and the named schema during autogeneration.
            connection.dialect.default_schema_name = 'public'
            _configure(
                schema=schema,
                metadata=metadata,
                connection=connection
            )
            with context.begin_transaction():
                context.run_migrations()
        engine.dispose()