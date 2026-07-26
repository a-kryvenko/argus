from app.schemas.response import error_response
from fastapi import APIRouter

router = APIRouter(prefix="/private/risk", tags=["risk"])

@router.get("/outlook")
def risk_outlook():
    return error_response(code="NOT_IMPLEMENTED", msg="Forecast not implemented.")

@router.get("/satelite-drag")
def risk_satelite_drag():
    return error_response(code="NOT_IMPLEMENTED", msg="Forecast not implemented.")

@router.get("/satelite-charge")
def risk_satelite_charge():
    return error_response(code="NOT_IMPLEMENTED", msg="Forecast not implemented.")
