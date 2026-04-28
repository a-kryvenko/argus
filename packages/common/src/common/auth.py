from datetime import datetime, timedelta
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = "change-this-to-a-very-long-random-string-in-production-2026"
ALGORITHM = "HS256"

security = HTTPBearer()

def create_access_token(contract_id: str = "demo-contract") -> str:
    expire = datetime.utcnow() + timedelta(days=30)
    payload = {
        "sub": contract_id,
        "exp": expire,
        "type": "private_access"
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]) -> str:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "private_access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        return payload.get("sub")
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )