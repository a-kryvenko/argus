from datetime import timedelta

from common.schemas.response import error_response, success_response
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import AwareDatetime
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.session import get_request_db_session
from clio.observations.indices import current_indices as read_current_indices
from clio.observations.normalized import load_normalized_observations
from clio.observations.summary import summary

router = APIRouter(prefix="/internal/v1/observations", tags=["observations"])


@router.get("/history")
async def observation_history(
    limit: int = Query(default=24, ge=1, le=8784),
    session: AsyncSession = Depends(get_request_db_session),
    start: AwareDatetime | None = Query(default=None, alias='from'),
    end: AwareDatetime | None = Query(default=None, alias='to'),
):
    """Return the most recent hourly observations, oldest first (up to 8784)."""
    bounds = {}
    if start is not None or end is not None:
        if start is None or end is None or start >= end:
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
    Stale inputs suppress trends.
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
    return success_response(await read_current_indices(session, selected, trends=trends))
