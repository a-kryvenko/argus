from app.db import get_db_session
from app.schemas.response import error_response, success_response
from app.services.sensor_observations import load_normalized_observations
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/public/observations", tags=["observations"])


@router.get("/history")
async def observation_history(
    limit: int = Query(default=24, ge=1, le=168),
    session: AsyncSession = Depends(get_db_session),
):
    """Return the most recent hourly observations, oldest first (up to 168)."""
    observations = await load_normalized_observations(session, limit=limit)
    return success_response(observations.model_dump())


@router.get("/latest")
async def latest_observations(
    session: AsyncSession = Depends(get_db_session),
):
    observations = await load_normalized_observations(session, limit=1)
    if not observations.points:
        return error_response(
            code="OBSERVATIONS_NOT_READY",
            msg="No observations found"
        )

    latest = observations.points[-1]
    return success_response(latest.model_dump())
