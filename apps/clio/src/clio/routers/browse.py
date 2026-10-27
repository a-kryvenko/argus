from typing import Literal

from common.schemas.response import success_response
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import AwareDatetime
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.session import get_request_db_session
from clio.observations.browse import browse

router = APIRouter(prefix='/internal/v1/observations')


@router.get('/browse')
async def observations(kind: Literal['raw', 'normalized'] = 'raw', page: int = Query(1, ge=1, le=100000),
    page_size: int = Query(50, ge=1, le=200), order: Literal['asc', 'desc'] = 'desc',
    start: AwareDatetime | None = None, end: AwareDatetime | None = None,
    metric: str | None = Query(None, max_length=16), db: AsyncSession = Depends(get_request_db_session)):
    if start and end and start > end:
        raise HTTPException(422, 'Start must precede end')
    return success_response(await browse(db, kind=kind, page=page, page_size=page_size,
                                         order=order, start=start, end=end, metric=metric))
