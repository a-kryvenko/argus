import os
from collections.abc import AsyncIterator

from sqlalchemy import URL
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_database_url(*, migration: bool = False) -> URL:
    """Build a safe connection URL from raw database credentials."""

    password_variable = "CLIO_MIGRATION_PASSWORD" if migration else "CLIO_DB_PASSWORD"
    required_variables = ("DB_NAME", password_variable)
    missing_variables = [name for name in required_variables if not os.getenv(name)]
    if missing_variables:
        missing = ", ".join(missing_variables)
        raise RuntimeError(f"Missing required database variables: {missing}")

    try:
        port = int(os.getenv("DB_PORT", "5432"))
    except ValueError as exc:
        raise RuntimeError("DB_PORT must be an integer") from exc

    return URL.create(
        drivername="postgresql+psycopg",
        username="argus_clio_migrator" if migration else "argus_clio",
        password=os.environ[password_variable],
        host=os.getenv("DB_HOST", "localhost"),
        port=port,
        database=os.environ["DB_NAME"],
    )


def get_engine() -> AsyncEngine:
    global _engine

    if _engine is None:
        _engine = create_async_engine(
            get_database_url(),
            pool_pre_ping=True,
            connect_args={"options": "-csearch_path=clio,pg_catalog,pg_temp"},
        )

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory

    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )

    return _session_factory


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Provide one transaction-aware session per request."""

    async with get_session_factory()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()

    _engine = None
    _session_factory = None
