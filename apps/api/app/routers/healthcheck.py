from fastapi import APIRouter
from app.schemas.response import success_response

router = APIRouter(tags=["ping"])

@router.get("/ping")
def ping():
    return success_response("pong")

