from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession
from argus_clio.db import get_db_session
from common.schemas.response import success_response
from argus_clio.services.observation_summary import summary

router = APIRouter(prefix='/internal/v1/observations', tags=['observations'])


@router.get('/summary')
async def observation_summary(response: Response, session: AsyncSession = Depends(get_db_session)):
    """Current values and observed trends with coverage; no forecast or impact score.

    Changes compare five-minute means one hour apart with at least 80% coverage
    in both windows and the last hour. Source changes and stale inputs suppress them.
    Southward Bz duration requires consecutive, unflagged samples from one spacecraft.
    """
    response.headers['Cache-Control'] = 'no-store'
    return success_response(await summary(session))
