from fastapi import APIRouter

from app.schemas.response import error_response

router = APIRouter(prefix="/private/probability", tags=["probability"])

@router.get("/bz")
def bz_probability():
    return error_response(code="NOT_IMPLEMENTED", msg="Forecast not implemented.")
