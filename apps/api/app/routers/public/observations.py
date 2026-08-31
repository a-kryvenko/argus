from app.db import get_db_session
from app.schemas.response import error_response, success_response
from app.services.sensor_observations import load_normalized_observations
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/public/observations", tags=["observations"])

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
