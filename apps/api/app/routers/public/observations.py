from fastapi import APIRouter

from app.schemas.response import SuccessResponse, ErrorResponse

router = APIRouter(prefix="/public/observations", tags=["observations"])

@router.get("/latest")
def latest_observations():
    return ErrorResponse(error="Forecast not implemented.")
