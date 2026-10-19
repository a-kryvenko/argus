from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession
from clio.db.session import get_db_session
from common.schemas.response import success_response
from clio.observations.summary import summary

router = APIRouter(prefix='/internal/v1/observations', tags=['observations'])


@router.get('/summary')
async def observation_summary(response: Response, session: AsyncSession = Depends(get_db_session)):
    """Current values and observed trends with coverage; no forecast or impact score.

    Solar-wind changes compare available samples in two five-minute windows one hour apart.
    Indices compare the last available values now and one hour ago.
    Source changes and stale inputs suppress trends.
    """
    response.headers['Cache-Control'] = 'no-store'
    return success_response(await summary(session))
