from app.schemas.response import success_response
from fastapi import APIRouter

router = APIRouter(tags=["ping"])

@router.get("/ping")
def ping():
    return success_response("pong")

