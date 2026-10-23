"""Bounded reads of native measurements and their hourly means."""
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import AwareDatetime, FiniteFloat
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.models.measurement import Measurement
from clio.db.session import get_db_session
from clio.routers.auth import require_service_token
from common.schemas.response import success_response

router = APIRouter(prefix='/internal/v1/observations', dependencies=[Depends(require_service_token)])


@router.get('/measurements')
async def measurements(
    start: AwareDatetime,
    end: AwareDatetime,
    metrics: str = Query(min_length=1, max_length=512),
    aggregation: Literal['raw', 'hourly'] = 'raw',
    min_value: FiniteFloat | None = None,
    max_value: FiniteFloat | None = None,
    exclude_value: FiniteFloat | None = None,
    session: AsyncSession = Depends(get_db_session),
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
    timestamp = Measurement.observed_at
    available = func.greatest(timestamp, Measurement.received_at, Measurement.interval_end)
    if aggregation == 'hourly':
        timestamp = func.date_trunc('hour', timestamp)
        statement = select(Measurement.metric, func.avg(Measurement.value).label('value'),
                           timestamp.label('observed_at'), func.max(available).label('available_at'))
        statement = statement.group_by(Measurement.metric, timestamp)
    else:
        statement = select(Measurement.metric, Measurement.value, timestamp,
                           available.label('available_at'))
    statement = statement.where(
        Measurement.metric.in_(selected), Measurement.observed_at >= start, Measurement.observed_at < end,
        or_(Measurement.quality.is_(None), Measurement.quality.not_in(['flagged', 'missing'])),
        Measurement.value > float('-inf'), Measurement.value < float('inf'),
    )
    if min_value is not None:
        statement = statement.where(Measurement.value >= min_value)
    if max_value is not None:
        statement = statement.where(Measurement.value < max_value)
    if exclude_value is not None:
        statement = statement.where(Measurement.value != exclude_value)
    rows = (await session.execute(statement.order_by(timestamp, Measurement.metric).limit(250_001))).all()
    if len(rows) > 250_000:
        raise HTTPException(503, 'Measurement range exceeds the supported batch size; request a smaller interval')
    return success_response({'schema_version': 1, 'items': [dict(row._mapping) for row in rows]})
