from fastapi import APIRouter

from app.schemas.response import SuccessResponse, ErrorResponse

router = APIRouter(prefix="/private/risk", tags=["risk"])

@router.get("/outlook")
def risk_outlook():
    return ErrorResponse(error="Forecast not implemented.")

@router.get("/satelite-drag")
def risk_satelite_drag():
    return ErrorResponse(error="Forecast not implemented.")

@router.get("/satelite-charge")
def risk_satelite_charge():
    return ErrorResponse(error="Forecast not implemented.")
