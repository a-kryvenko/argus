"""The numeric Clio contract: last observations and bounded metric history."""
from datetime import UTC, datetime, timedelta
from typing import Literal

from common.schemas.response import success_response
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import AwareDatetime
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.session import get_request_db_session
from clio.observations import metrics as reads
from clio.observations.errors import ObservationUnavailable

router = APIRouter(prefix='/internal/v1/observations', tags=['observations'])


def selected_metrics(metrics: str | None = None):
    selected = list(reads.METRICS) if metrics is None else list(dict.fromkeys(m.strip() for m in metrics.split(',')))
    if not selected or any(m not in reads.METRICS for m in selected):
        raise HTTPException(422, 'Unknown observation metric')
    return selected


@router.get('/latest')
async def latest(response: Response, metrics: list[str] = Depends(selected_metrics),
                 as_of: AwareDatetime | None = None, session: AsyncSession = Depends(get_request_db_session)):
    now = datetime.now(UTC)
    if as_of is not None and as_of > now:
        raise HTTPException(422, 'as_of must not be in the future')
    response.headers['Cache-Control'] = 'no-store'
    return success_response(await reads.latest(session, metrics, as_of or now))


@router.get('/history')
async def history(response: Response, start: AwareDatetime = Query(alias='from'),
                  end: AwareDatetime = Query(alias='to'),
                  interval: Literal['1m', '5m', '1h'] = '1h',
                  metrics: list[str] = Depends(selected_metrics), as_of: AwareDatetime | None = None,
                  session: AsyncSession = Depends(get_request_db_session)):
    now = datetime.now(UTC)
    cutoff = as_of or now
    if not timedelta(0) < end-start <= timedelta(days=366) or cutoff > now:
        raise HTTPException(422, 'Use a past interval of at most 366 days and a past as_of')
    response.headers['Cache-Control'] = 'no-store'
    try:
        data = await reads.history(session, metrics, start.astimezone(UTC), end.astimezone(UTC), interval, cutoff)
    except ObservationUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    return success_response(data)
