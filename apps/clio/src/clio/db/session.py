from collections.abc import AsyncIterator
from functools import lru_cache

from common.db.session import get_database_url
from sqlalchemy import create_engine

from common.db.session import get_db_session as _get_db_session
from sqlalchemy.ext.asyncio import AsyncSession


def get_db_session():
    return _get_db_session('clio')

async def get_request_db_session() -> AsyncIterator[AsyncSession]:
    async with get_db_session() as session:
        yield session


@lru_cache(maxsize=1)
def get_sync_engine():
    return create_engine(get_database_url('CLIO'), pool_pre_ping=True,
        connect_args={'options': '-csearch_path=clio,public,pg_catalog,pg_temp'})


def dispose_sync_engine():
    if get_sync_engine.cache_info().currsize:
        get_sync_engine().dispose()
        get_sync_engine.cache_clear()
