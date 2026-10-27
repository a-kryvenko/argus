from clio.db.session import get_request_db_session
from common.schemas.response import error_response, success_response
from clio.observations.normalized import load_normalized_observations
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from datetime import UTC, datetime, timedelta
from sqlalchemy import select, tuple_
from clio.db.models.measurement import Measurement
from clio.observations.summary import summary, value_change
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/internal/v1/observations", tags=["observations"])


@router.get("/history")
async def observation_history(
    limit: int = Query(default=24, ge=1, le=8784),
    session: AsyncSession = Depends(get_request_db_session),
    start: datetime | None = Query(default=None, alias='from'),
    end: datetime | None = Query(default=None, alias='to'),
):
    """Return the most recent hourly observations, oldest first (up to 8784)."""
    bounds = {}
    if start is not None or end is not None:
        if start is None or end is None or start.tzinfo is None or end.tzinfo is None or start >= end:
            raise HTTPException(422, 'from and to must define a positive timezone-aware interval')
        bounds = {'since': start, 'until': end - timedelta(microseconds=1)}
    observations = await load_normalized_observations(session, limit=limit, **bounds)
    return success_response(observations)


@router.get("/latest")
async def latest_observations(
    session: AsyncSession = Depends(get_request_db_session),
):
    observations = await load_normalized_observations(session, limit=1)
    if not observations.points:
        return error_response(
            code="OBSERVATIONS_NOT_READY",
            msg="No observations found"
        )

    latest = observations.points[-1]
    return success_response(latest)


@router.get('/summary')
async def observation_summary(response: Response, session: AsyncSession = Depends(get_request_db_session)):
    """Current values and observed trends with coverage; no forecast or impact score.

    Solar-wind changes compare available samples in two five-minute windows one hour apart.
    Indices compare the last available values now and one hour ago.
    Source changes and stale inputs suppress trends.
    """
    response.headers['Cache-Control'] = 'no-store'
    return success_response(await summary(session))


@router.get('/current-indices')
async def current_indices(
    metrics: str = Query(default='ap,f10_7,s10,m10,y10'),
    trends: bool = Query(default=False),
    session: AsyncSession = Depends(get_request_db_session),
):
    selected = list(dict.fromkeys(value.strip() for value in metrics.split(',')))
    if any(value not in ('ap', 'f10_7', 's10', 'm10', 'y10') for value in selected):
        raise HTTPException(422, 'Unknown index')
    # Legacy stored S10/M10/Y10 estimates are not source observations.
    now = datetime.now(UTC)
    statement = (select(Measurement)
                 .where(Measurement.metric.in_(selected), Measurement.observed_at <= now,
                        tuple_(Measurement.metric, Measurement.source_product).in_([
                            ('ap', 'swpc.kp'), ('f10_7', 'swpc.f107'), ('f10_7', 'gfz.f107'),
                        ]))
                 .distinct(Measurement.metric)
                 .order_by(Measurement.metric, Measurement.observed_at.desc()))
    records = (await session.execute(statement)).scalars().all()
    series = dict.fromkeys(selected)
    for record in records:
        point = {'observed_at': record.observed_at, 'value': record.value,
                 'quality': record.quality or ('missing' if record.value is None else None)}
        if record.interval_end is not None:
            point['interval_end'] = record.interval_end
        series[record.metric] = point
    data = {'series': series}
    if trends:
        previous_records = (await session.execute(statement.where(
            Measurement.observed_at <= now-timedelta(hours=1)))).scalars().all()
        previous = {record.metric: {'value': record.value, 'quality': record.quality} for record in previous_records}
        data['changes_1h'] = {metric: value_change(series[metric], previous.get(metric)) for metric in selected}
    return success_response(data)
