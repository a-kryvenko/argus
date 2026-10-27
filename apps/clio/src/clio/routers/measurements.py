"""Bounded reads of native measurements and their hourly means."""
from datetime import UTC, datetime, timedelta
from typing import Literal

from common.schemas.response import success_response
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import AwareDatetime, FiniteFloat
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.session import get_request_db_session
from clio.observations.errors import ObservationUnavailable
from clio.observations.measurements import read_measurements

router = APIRouter(prefix='/internal/v1/observations')


@router.get('/measurements')
async def measurements(
    start: AwareDatetime,
    end: AwareDatetime,
    metrics: str = Query(min_length=1, max_length=512),
    aggregation: Literal['raw', 'hourly'] = 'raw',
    min_value: FiniteFloat | None = None,
    max_value: FiniteFloat | None = None,
    exclude_value: FiniteFloat | None = None,
    session: AsyncSession = Depends(get_request_db_session),
):
    """Read [start, end); hourly means include partial buckets, without filling gaps.

    Null, nonfinite and flagged/missing samples are excluded. Optional value bounds
    apply before aggregation; max_value is exclusive. Oversized reads fail explicitly.
    """
    if not timedelta(0) < end - start <= timedelta(days=366) or end > datetime.now(UTC):
        raise HTTPException(422, 'Interval must be positive, at most 366 days and end in the past')
    selected = list(dict.fromkeys(value.strip() for value in metrics.split(',')))
    if any(not metric or len(metric) > 16 for metric in selected):
        raise HTTPException(422, 'Invalid metric')
    if min_value is not None and max_value is not None and min_value >= max_value:
        raise HTTPException(422, 'min_value must be less than max_value')
    try:
        data = await read_measurements(session, start=start, end=end, metrics=selected, aggregation=aggregation,
                                       min_value=min_value, max_value=max_value, exclude_value=exclude_value)
    except ObservationUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    return success_response(data)
