"""Authenticated on-demand impact calculations; no forecast generation."""
import os
import secrets

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from common.schemas.leo_drag import LeoDragAssessment, LeoDragRequest
from argus_intelligence.drag import assess_drag, DragDomainError, DragNotReadyError

app = FastAPI(title='Intelligence impact service', version='1')
security = HTTPBearer(auto_error=False)


def require_service_token(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    expected = os.getenv('INTELLIGENCE_SERVICE_TOKEN')
    if not expected:
        raise HTTPException(503, 'Impact service is not configured')
    if credentials is None or not secrets.compare_digest(credentials.credentials.encode(), expected.encode()):
        raise HTTPException(401, 'Invalid service credentials')


@app.post('/internal/v1/risks/leo-drag', response_model=LeoDragAssessment,
          dependencies=[Depends(require_service_token)])
def leo_drag(request: LeoDragRequest):
    try:
        return assess_drag(request)
    except DragNotReadyError:
        raise HTTPException(503, 'Atmospheric density or impact backend is not ready') from None
    except DragDomainError as exc:
        raise HTTPException(422, str(exc)) from None


@app.get('/health/live')
def live():
    return {'service': 'intelligence', 'status': 'ok'}


@app.get('/health/ready')
def ready():
    if not all(os.getenv(key) for key in ('INTELLIGENCE_SERVICE_TOKEN', 'FORECASTS_URL', 'FORECASTS_SERVICE_TOKEN')):
        raise HTTPException(503, 'Impact service is not configured')
    try:
        from intelligence_core.api import calculate_drag
    except ImportError:
        raise HTTPException(503, 'Impact backend is not installed') from None
    return {'service': 'intelligence', 'status': 'ok', 'density_readiness': 'checked_per_request'}
