from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_db_session
from app.schemas.response import success_response
from app.services.observation_summary import summary

router = APIRouter(prefix='/public/observations', tags=['observations'])


@router.get('/summary')
async def observation_summary(response: Response, session: AsyncSession = Depends(get_db_session)):
    """Current values and observed trends with coverage; no forecast or impact score.

    Changes compare five-minute means one hour apart with at least 80% coverage
    in both windows and the last hour. Source changes and stale inputs suppress them.
    Southward Bz duration requires consecutive, unflagged samples from one spacecraft.
    """
    response.headers['Cache-Control'] = 'no-store'
    return success_response(await summary(session))
