import os
import secrets
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)


def require_service_token(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    expected = os.getenv('OBSERVATIONS_SERVICE_TOKEN')
    if not expected:
        raise HTTPException(503, 'Observation read service is not configured')
    if credentials is None or not secrets.compare_digest(credentials.credentials.encode(), expected.encode()):
        raise HTTPException(401, 'Invalid service credentials')


