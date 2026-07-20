from fastapi import APIRouter

from app.schemas.response import success_response, error_response
from forecast.data_pipelines.live import get_live_observations

router = APIRouter(prefix="/public/observations", tags=["observations"])

@router.get("/latest")
def latest_observations():
    observations = get_live_observations()
    latest = observations.points[-1] or None
    if latest is None:
        return error_response(
            code="OBSERVATIONS_NOT_READY",
            msg="No observations found"
        )
    
    return success_response(latest.model_dump())
