from fastapi import APIRouter
from app.schemas.response import SuccessResponse

router = APIRouter(tags=["healthcheck"])

@router.get("/healthcheck")
def get_healthcheck_status():
    return SuccessResponse(data={"service_status": "up"})

