from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db_session
from app.schemas.response import success_response
from app.services import solar_wind

router = APIRouter(prefix="/public/observations/solar-wind", tags=["observations"])


def selected_metrics(metrics: str | None = Query(default=None, description="Comma-separated bx,by,bz,bt,v,n,t")) -> list[str]:
    selected = list(solar_wind.METADATA) if metrics is None else list(dict.fromkeys(m.strip() for m in metrics.split(',')))
    if not selected or any(metric not in solar_wind.METADATA for metric in selected):
        raise HTTPException(422, "Unknown metric. Supported: bx,by,bz,bt,v,n,t")
    return selected


@router.get("/latest")
async def latest(
    response: Response,
    metrics: list[str] = Depends(selected_metrics),
    session: AsyncSession = Depends(get_db_session),
):
    """Latest active-source samples; freshness is based on measurement time.

    Missing/flagged values never silently fall back to an older good sample.
    Quality 'unverified' means Argus has not independently validated the sample.
    """
    response.headers["Cache-Control"] = "no-store"
    return success_response(await solar_wind.latest(session, metrics))


@router.get("/history")
async def history(
    response: Response,
    start: datetime | None = Query(default=None, alias="from", description="Inclusive UTC timestamp; default last 24 hours"),
    end: datetime | None = Query(default=None, alias="to", description="Exclusive UTC timestamp; default now"),
    metrics: list[str] = Depends(selected_metrics),
    session: AsyncSession = Depends(get_db_session),
):
    """Native one-minute samples, oldest first, without filling or resampling.

    Maximum range: seven days. Request adjacent intervals for longer history.
    Omitted timestamps are gaps; explicit missing values are null.
    """
    end = end or datetime.now(UTC)
    start = start or end - timedelta(hours=24)
    if start.tzinfo is None or end.tzinfo is None:
        raise HTTPException(422, "from and to must include a timezone")
    start, end = start.astimezone(UTC), end.astimezone(UTC)
    if not timedelta(0) < end - start <= timedelta(days=7):
        raise HTTPException(422, "History interval must be positive and at most seven days")
    response.headers["Cache-Control"] = "no-store"
    return success_response(await solar_wind.history(session, metrics, start, end))
