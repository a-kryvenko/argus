from fastapi import APIRouter
from core.auth import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])

@router.get("/token")
async def issue_token(contract_id: str = "demo-contract"):
    token = create_access_token(contract_id)
    return {"access_token": token, "token_type": "bearer", "expires_in_days": 30}