from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession
from argus_clio.db import get_db_session
from common.schemas.response import success_response
from argus_clio.services.collection_status import source_status

router = APIRouter(prefix='/internal/v1/observations', tags=['observations'])


@router.get('/status')
async def collection_status(response: Response, session: AsyncSession = Depends(get_db_session)):
    """Source diagnostics: attempts, successful responses, storage outcomes and data age.

    Status priority: stalled/overdue collection, errors, then delayed/missing data.
    No recent attempts indicates missing progress; it does not inspect processes.
    A historical last error is retained after recovery; consecutive_failures resets.
    """
    response.headers['Cache-Control'] = 'no-store'
    return success_response(await source_status(session))


@router.get('/monitoring')
async def monitoring(response: Response, session: AsyncSession = Depends(get_db_session)):
    from argus_clio.services.monitoring import monitoring_status
    response.headers['Cache-Control'] = 'no-store'
    return success_response(await monitoring_status(session))
