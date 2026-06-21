from fastapi import APIRouter

from app.schemas.response import SuccessResponse, ErrorResponse
from forecast.data_pipelines.live import get_live_observations

router = APIRouter(prefix="/public/observations", tags=["observations"])

@router.get("/latest")
def latest_observations():
    observations = get_live_observations()
    latest = observations.points[-1] or None
    if latest is None:
        return ErrorResponse(error="No observations found")
    
    return SuccessResponse(data=latest.model_dump())
