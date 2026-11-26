import asyncio
import logging

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from api.db.session import get_db_session


router = APIRouter()
logger = logging.getLogger(__name__)
READINESS_QUERY = text('''
    SELECT u.password_hash, g.permissions, m.group_name, s.expires_at, a.window
    FROM dashboard_user u, dashboard_group g, dashboard_membership m,
         dashboard_session s, dashboard_login_attempt a
    LIMIT 0
''')


@router.get('/health/live')
async def live():
    return {'service': 'api', 'status': 'ok'}


@router.get('/health/ready')
async def ready():
    try:
        async with asyncio.timeout(5):
            async with get_db_session() as db:
                await db.execute(READINESS_QUERY)
    except Exception as exc:
        logger.warning('Dashboard readiness failed: %s', type(exc).__name__)
        raise HTTPException(503, 'API storage is not ready') from None
    return {'service': 'api', 'status': 'ok'}
