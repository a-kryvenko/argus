from app.schemas.response import error_response, success_response
from clio.sensors import Sensors
from fastapi import APIRouter

router = APIRouter(prefix="/public/observations", tags=["observations"])

@router.get("/latest")
def latest_observations():
    observations = Sensors.get_live_observations_frame()
    latest = observations.points[-1] or None
    if latest is None:
        return error_response(
            code="OBSERVATIONS_NOT_READY",
            msg="No observations found"
        )
    
    return success_response(latest.model_dump())
