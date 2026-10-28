import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

_session_factories: dict[
    str,
    async_sessionmaker[AsyncSession],
] = {}

_engines: dict[str, AsyncEngine] = {}

def database_parameters(prefix: str) -> dict:
    """An empty prefix selects the administrative connection to postgres."""
    base = f'{prefix}_DB_' if prefix else 'DB_'

    fields = ('NAME', 'USER', 'PASSWORD') if prefix else ('USER', 'PASSWORD')
    required = [base + field for field in fields]
    missing = [key for key in required if not os.getenv(key)]

    if missing:
        raise ValueError('Missing database settings: ' + ', '.join(missing))

    host = os.getenv(base + 'HOST', 'localhost')
    if not host:
        raise ValueError(base + 'HOST must not be empty')

    try:
        port = int(os.getenv(base + 'PORT', '5432'))
        if not 1 <= port <= 65535:
            raise ValueError()
    except ValueError:
        raise ValueError(base + 'PORT must be an integer between 1 and 65535') from None

    return {
        'host': host,
        'port': port,
        'database': os.environ[base + 'NAME'] if prefix else 'postgres',
        'username': os.environ[base + 'USER'],
        'password': os.environ[base + 'PASSWORD']
    }


def get_database_url(service: str):
    from sqlalchemy import URL

    try:
        parameters = database_parameters(service.upper())
    except ValueError as exc:
        raise RuntimeError(str(exc)) from None
    
    return URL.create('postgresql+psycopg', **parameters)

def _get_engine(service: str) -> AsyncEngine:
    if service not in _engines:
        _engines[service] = create_async_engine(
            get_database_url(service),
            pool_pre_ping=True,
            connect_args={"options": f"-csearch_path={service},public,pg_catalog,pg_temp"},
        )

    return _engines[service]

def _get_session_factory(service: str) -> async_sessionmaker[AsyncSession]:
    if service not in _session_factories:
        _session_factories[service] = async_sessionmaker(
            bind=_get_engine(service),
            class_=AsyncSession,
            expire_on_commit=False,
        )

    return _session_factories[service]


@asynccontextmanager
async def get_db_session(service: str) -> AsyncGenerator[AsyncSession, None]:
    # Keep one connection across commits until the session context exits.
    async with _get_engine(service).connect() as connection:
        async with _get_session_factory(service)(bind=connection) as session:
            yield session


async def dispose_engine() -> None:
    global _engines, _session_factories

    for _engine in _engines.values():
        await _engine.dispose()

    _engines = {}
    _session_factories = {}
