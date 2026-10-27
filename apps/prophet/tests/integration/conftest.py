import pytest
from common.db.session import dispose_engine
from tests.support.database import database


@pytest.fixture(autouse=True)
async def database_runtime(database, anyio_backend):
    try:
        yield
    finally:
        await dispose_engine()
