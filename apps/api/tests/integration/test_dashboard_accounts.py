"""Exercise dashboard storage under the deployed PostgreSQL schema."""
import asyncio
from datetime import UTC, datetime, timedelta
import subprocess
import sys

from sqlalchemy import text

from api.db.session import get_db_session
from api.services.authentication import LOGIN_LIMIT, LOGIN_WINDOW, reserve_login_attempt
from common.db import session
from tests.support.database import ROOT, database


def test_login_window_and_connection_lifetime(database, monkeypatch):
    _, _, environment = database
    for key in ('HOST', 'PORT', 'NAME', 'USER', 'PASSWORD'):
        setting = f'API_DB_{key}'
        monkeypatch.setenv(setting, environment[setting])
    subprocess.run([sys.executable, '-m', 'alembic', '-c', str(ROOT / 'apps/api/alembic.ini'), 'upgrade', 'head'],
                   env=environment, cwd=ROOT, check=True, timeout=60)

    async def check():
        await session.dispose_engine()
        engine = session._get_engine('api')
        current = datetime(2026, 11, 25, 12, 15, tzinfo=UTC)
        try:
            async with get_db_session() as db:
                assert engine.pool.checkedout() == 0
                await db.execute(text('SELECT 1'))
                assert engine.pool.checkedout() == 1
                await db.commit()
                assert engine.pool.checkedout() == 0
                assert await reserve_login_attempt(db, 'test-reader', current) == (1, current)
                assert await reserve_login_attempt(db, 'test-reader', current - timedelta(minutes=1)) == (2, current)
                for _ in range(LOGIN_LIMIT + 2):
                    count, window = await reserve_login_attempt(db, 'test-reader', current)
                assert count == LOGIN_LIMIT + 1 and window == current
                following = current + LOGIN_WINDOW
                assert await reserve_login_attempt(db, 'test-reader', following) == (1, following)
                await db.commit()
                assert engine.pool.checkedout() == 0
        finally:
            await session.dispose_engine()

    asyncio.run(check())
