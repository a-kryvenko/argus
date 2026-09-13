from datetime import datetime
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from argus_clio.db import get_db_session
from argus_clio.db.models import Measurement, NormalizedObservation
from common.schemas.response import success_response

router = APIRouter(prefix='/internal/v1/observations')


@router.get('/browse')
async def observations(kind: Literal['raw', 'normalized'] = 'raw', page: int = Query(1, ge=1, le=100000),
    page_size: int = Query(50, ge=1, le=200), order: Literal['asc', 'desc'] = 'desc',
    start: datetime | None = None, end: datetime | None = None,
    metric: str | None = Query(None, max_length=16), db: AsyncSession = Depends(get_db_session)):
    for value in [start, end]:
        if value is not None and value.tzinfo is None:
            raise HTTPException(422, 'Dates must include a timezone')
    if start and end and start > end:
        raise HTTPException(422, 'Start must precede end')
    model = Measurement if kind == 'raw' else NormalizedObservation
    filters = []
    if start:
        filters.append(model.observed_at >= start)
    if end:
        filters.append(model.observed_at <= end)
    if metric and kind == 'raw':
        filters.append(Measurement.metric == metric)
    ordering = [model.observed_at.asc() if order == 'asc' else model.observed_at.desc()]
    if kind == 'raw':
        ordering.append(Measurement.id.asc() if order == 'asc' else Measurement.id.desc())
    rows = (await db.scalars(select(model).where(*filters).order_by(*ordering).offset((page-1)*page_size).limit(page_size))).all()
    columns = [c.name for c in model.__table__.columns]
    return success_response({'columns': columns, 'items': [{c: getattr(row, c) for c in columns} for row in rows],
        'total': await db.scalar(select(func.count()).select_from(model).where(*filters)), 'page': page, 'page_size': page_size})


