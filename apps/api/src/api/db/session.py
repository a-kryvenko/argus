from collections.abc import AsyncIterator

from common.db.session import get_db_session as _get_db_session
from sqlalchemy.ext.asyncio import AsyncSession


def get_db_session():
    return _get_db_session('api')

async def get_request_db_session() -> AsyncIterator[AsyncSession]:
    async with get_db_session() as session:
        yield session
