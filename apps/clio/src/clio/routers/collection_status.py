from common.schemas.response import success_response
from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.session import get_request_db_session
from clio.observations.status import observation_status

router = APIRouter(prefix='/internal/v1/observations', tags=['observations'])


@router.get('/status')
async def collection_status(response: Response, session: AsyncSession = Depends(get_request_db_session)):
    """Availability and freshness of stored observations; no source requests."""
    response.headers['Cache-Control'] = 'no-store'
    return success_response(await observation_status(session))
