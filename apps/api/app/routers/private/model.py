from fastapi import APIRouter

from app.schemas.response import SuccessResponse, ErrorResponse

router = APIRouter(prefix="/private/model", tags=["risk"])

@router.get("/quality")
def risk_outlook():
    return ErrorResponse(error="Forecast not implemented.")