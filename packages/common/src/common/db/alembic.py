from alembic import context
from sqlalchemy import create_engine, pool, text


def _configure(schema: str, metadata, **kwargs):
    def _include_name(name, type_, parent_names):
        if type_ == 'table':
            return name not in ('alembic_version', 'legacy_alembic_version')
        
        return True

    context.configure(
        target_metadata=metadata,
        compare_type=True,
        include_schemas=False,
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
            context.execute(f"SET search_path TO {schema}, public, pg_catalog, pg_temp")
            context.run_migrations()
    else:
        engine = create_engine(
            url=url,
            poolclass=pool.NullPool,
            connect_args={'options': f"-csearch_path={schema},public,pg_catalog,pg_temp -clock_timeout=5000 -cstatement_timeout=300000"}
        )
        with engine.connect() as connection:
            connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            connection.commit()
            # Unqualified model names reflect against the current search-path schema.
            connection.dialect.default_schema_name = connection.scalar(text('SELECT current_schema()'))
            connection.commit()
            _configure(
                schema=schema,
                metadata=metadata,
                connection=connection
            )
            with context.begin_transaction():
                context.run_migrations()
        engine.dispose()
