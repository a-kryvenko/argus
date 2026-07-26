from app.schemas.response import error_response
from fastapi import APIRouter

router = APIRouter(prefix="/private/model", tags=["risk"])

@router.get("/quality")
def risk_outlook():
    return error_response(
        code="NOT_IMPLEMENTED",
        msg="Forecast not implemented."
    )