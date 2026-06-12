from fastapi import APIRouter

from app.schemas.response import SuccessResponse, ErrorResponse

router = APIRouter(prefix="/private/forecast", tags=["forecast-private"])

@router.get("/imf")
def imf_forecast():
    return ErrorResponse(error="Forecast not implemented.")

@router.get("/plasma")
def plasma_forecast():
    return ErrorResponse(error="Forecast not implemented.")
