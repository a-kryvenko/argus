from fastapi import APIRouter

from app.schemas.response import SuccessForecastResponse, ErrorResponse

router = APIRouter(prefix="/private/probability", tags=["probability"])

@router.get("/bz")
def bz_probability():
    return ErrorResponse(error="Forecast not implemented.")
