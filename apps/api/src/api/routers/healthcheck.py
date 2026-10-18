from common.schemas.response import success_response
import asyncio
from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from api.db.session import get_session_factory

router = APIRouter(tags=["ping"])

@router.get("/ping")
def ping():
    return success_response("pong")


@router.get('/health/live', include_in_schema=False)
def live():
    return {'service': 'api', 'status': 'ok'}


@router.get('/health/ready', include_in_schema=False)
async def ready():
    try:
        async with get_session_factory()() as db:
            await asyncio.wait_for(db.execute(text('SELECT id FROM dashboard_user LIMIT 1')), timeout=5)
    except Exception:
        raise HTTPException(503, 'API storage is not ready') from None
    return {'service': 'api', 'status': 'ok'}

