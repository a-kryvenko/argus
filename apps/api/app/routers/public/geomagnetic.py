from datetime import UTC, datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Response
from app.services.observations_client import read_observations

router = APIRouter(prefix='/public/observations/geomagnetic', tags=['observations'])


@router.get('/latest')
async def latest(response: Response):
    """Latest native Kp/Dst intervals, with independent quality and publication lag."""
    response.headers['Cache-Control'] = 'no-store'
    return await read_observations('geomagnetic/latest')


@router.get('/history')
async def history(response: Response,
                  start: datetime | None = Query(default=None, alias='from'),
                  end: datetime | None = Query(default=None, alias='to')):
    """Native intervals overlapping [from,to); default 24h, maximum 31 days.

    Intervals are not clipped, rounded, interpolated or expanded into hourly Kp.
    An overlapping interval may appear in adjacent requests: deduplicate by its start.
    """
    end = end or datetime.now(UTC)
    start = start or end-timedelta(hours=24)
    if start.tzinfo is None or end.tzinfo is None:
        raise HTTPException(422, 'from and to must include a timezone')
    start, end = start.astimezone(UTC), end.astimezone(UTC)
    if not timedelta(0) < end-start <= timedelta(days=31):
        raise HTTPException(422, 'Interval must be positive and at most 31 days')
    response.headers['Cache-Control'] = 'no-store'
    return await read_observations('geomagnetic/history', {'from': start, 'to': end})
