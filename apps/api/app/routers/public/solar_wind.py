from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from app.services.observations_client import read_observations

METRICS = ("bx", "by", "bz", "bt", "v", "n", "t")
from typing import Literal

router = APIRouter(prefix="/public/observations/solar-wind", tags=["observations"])


def selected_metrics(metrics: str | None = Query(default=None, description="Comma-separated bx,by,bz,bt,v,n,t")) -> list[str]:
    selected = list(METRICS) if metrics is None else list(dict.fromkeys(m.strip() for m in metrics.split(',')))
    if not selected or any(metric not in METRICS for metric in selected):
        raise HTTPException(422, "Unknown metric. Supported: bx,by,bz,bt,v,n,t")
    return selected


@router.get("/latest")
async def latest(
    response: Response,
    metrics: list[str] = Depends(selected_metrics),
):
    """Latest active-source samples; freshness is based on measurement time.

    Missing/flagged values never silently fall back to an older good sample.
    Quality 'unverified' means Argus has not independently validated the sample.
    """
    response.headers["Cache-Control"] = "no-store"
    return await read_observations('solar-wind/latest', {'metrics': ','.join(metrics)})


@router.get("/history")
async def history(
    response: Response,
    start: datetime | None = Query(default=None, alias="from", description="Inclusive UTC timestamp; default last 24 hours"),
    end: datetime | None = Query(default=None, alias="to", description="Exclusive UTC timestamp; default now"),
    resolution: Literal["1m", "5m", "1h", "auto"] = Query(default="1m"),
    metrics: list[str] = Depends(selected_metrics),
):
    """Stored native samples or complete UTC aggregate windows.

    Limits: 1m seven days, 5m 31 days, 1h 366 days. Auto: 1m through 24h,
    5m through seven days, otherwise 1h. Aggregate left edge may precede from;
    current partial buckets are excluded. No source requests or recalculation.
    Omitted timestamps are gaps; explicit missing values are null.
    """
    end = end or datetime.now(UTC)
    start = start or end - timedelta(hours=24)
    if start.tzinfo is None or end.tzinfo is None:
        raise HTTPException(422, "from and to must include a timezone")
    start, end = start.astimezone(UTC), end.astimezone(UTC)
    if resolution == 'auto':
        resolution = '1m' if end-start <= timedelta(days=1) else '5m' if end-start <= timedelta(days=7) else '1h'
    days = {'1m': 7, '5m': 31, '1h': 366}[resolution]
    if not timedelta(0) < end - start <= timedelta(days=days):
        raise HTTPException(422, f"History interval must be positive and at most {days} days for {resolution}")
    response.headers["Cache-Control"] = "no-store"
    return await read_observations('solar-wind/history', {
        'from': start, 'to': end, 'resolution': resolution, 'metrics': ','.join(metrics),
    })
