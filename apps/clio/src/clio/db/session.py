from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_database_url():
    """Runtime and Alembic share credentials; passwords remain raw strings."""
    from sqlalchemy import URL
    from common.database import database_parameters
    try:
        parameters = database_parameters('CLIO')
    except ValueError as exc:
        raise RuntimeError(str(exc)) from None
    return URL.create('postgresql+psycopg', **parameters)


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
