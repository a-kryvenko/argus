"""Authenticated on-demand impact calculations; no forecast generation."""
import os
import secrets
from contextlib import asynccontextmanager

from common.db.session import dispose_engine
from intelligence.db.session import get_db_session
from sqlalchemy import text

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from common.schemas.leo_drag import LeoDragAssessment, LeoDragRequest
from intelligence.drag import assess_drag, DragDomainError, DragNotReadyError
from intelligence.monitor_routes import router as monitor_router
from intelligence.density_routes import router as density_router


@asynccontextmanager
async def lifespan(_):
    try:
        yield
    finally:
        await dispose_engine()


app = FastAPI(title='Intelligence impact service', version='1', lifespan=lifespan)
security = HTTPBearer(auto_error=False)


def require_service_token(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    expected = os.getenv('INTELLIGENCE_SERVICE_TOKEN')
    if not expected:
        raise HTTPException(503, 'Impact service is not configured')
    if credentials is None or not secrets.compare_digest(credentials.credentials.encode(), expected.encode()):
        raise HTTPException(401, 'Invalid service credentials')


app.include_router(monitor_router, dependencies=[Depends(require_service_token)])
app.include_router(density_router, dependencies=[Depends(require_service_token)])


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
async def ready():
    if not all(os.getenv(key) for key in ('INTELLIGENCE_SERVICE_TOKEN', 'FORECASTS_URL', 'FORECASTS_SERVICE_TOKEN')):
        raise HTTPException(503, 'Impact service is not configured')
    try:
        from intelligence_core.api import assess_density, describe_mode, summarize_monitor, prioritize_monitors, prepare_operator_chart, evaluate_drag_events, assess_density_exposure, summarize_density_advisory, evaluate_advisory_events, reuse_density_advisory
    except ImportError:
        raise HTTPException(503, 'Impact backend is not installed') from None
    try:
        async with get_db_session() as session:
            await session.execute(text('SET LOCAL statement_timeout=3000'))
            await session.execute(text('SELECT id FROM drag_profile LIMIT 1'))
    except Exception:
        raise HTTPException(503, 'Impact storage is not ready') from None
    return {'service': 'intelligence', 'status': 'ok', 'density_readiness': 'checked_per_request'}
