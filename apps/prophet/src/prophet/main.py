"""Internal HTTP reads of committed Prophet product releases."""
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging
import os
import secrets
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from common.config import get_config
from common.schemas.forecast_release import ForecastRelease, PRODUCT_ARTIFACTS
from common.schemas.forecast_status import ForecastStatus
from common.schemas.forecast_verification import ForecastVerification
from common.db.session import dispose_engine
from prophet.db.session import get_db_session
from sqlalchemy import text
from prophet.services.releases.publication import read_release, ReleaseNotFound

get_config()


@asynccontextmanager
async def lifespan(_) -> AsyncGenerator[None, None]:
    try:
        yield
    finally:
        await dispose_engine()


app = FastAPI(title='Prophet forecast service', version='1', lifespan=lifespan)
security = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


def require_service_token(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    expected = os.getenv('FORECASTS_SERVICE_TOKEN')
    if not expected:
        raise HTTPException(503, 'Forecast read service is not configured')
    if credentials is None or not secrets.compare_digest(credentials.credentials.encode(), expected.encode()):
        raise HTTPException(401, 'Invalid service credentials')


async def load(product, release_id=None):
    if product not in PRODUCT_ARTIFACTS:
        raise HTTPException(404, 'Unknown forecast product')
    try:
        return await read_release(product, release_id)
    except ReleaseNotFound:
        raise HTTPException(404 if release_id else 503, 'Forecast release is not available') from None
    except Exception:
        logger.exception('Forecast release read failed')
        raise HTTPException(503, 'Forecast storage is not ready') from None


@app.get('/internal/v1/forecasts/{product}/latest', response_model=ForecastRelease,
         dependencies=[Depends(require_service_token)])
async def latest(product: str):
    return await load(product)


@app.get('/internal/v1/forecasts/{product}/releases/{release_id}', response_model=ForecastRelease,
         dependencies=[Depends(require_service_token)])
async def historical(product: str, release_id: UUID):
    return await load(product, release_id)


@app.get('/internal/v1/forecasts/{product}/status', response_model=ForecastStatus,
         dependencies=[Depends(require_service_token)])
async def status(product: str):
    if product not in PRODUCT_ARTIFACTS:
        raise HTTPException(404, 'Unknown forecast product')
    from prophet.services.releases.status import product_status
    try:
        return await product_status(product)
    except Exception:
        logger.exception('Forecast status read failed')
        raise HTTPException(503, 'Forecast storage is not ready') from None


@app.get('/health/live')
def live():
    return {'service': 'prophet', 'status': 'ok'}


@app.get('/health/ready')
async def ready():
    if not os.getenv('FORECASTS_SERVICE_TOKEN'):
        raise HTTPException(503, 'Forecast read service is not configured')
    try:
        async with get_db_session() as session:
            await session.execute(text('SET LOCAL statement_timeout=3000'))
            await session.execute(text('SELECT release_id FROM prophet.current_forecast LIMIT 1'))
    except Exception:
        raise HTTPException(503, 'Forecast storage is not ready') from None
    return {'service': 'prophet', 'status': 'ok'}


@app.get('/internal/v1/forecasts/{product}/verification',
         response_model=ForecastVerification,
         dependencies=[Depends(require_service_token)])
async def verification(product: str):
    if product == 'solar-wind-events':
        raise HTTPException(404, 'Event verification is not available yet')
    if product not in PRODUCT_ARTIFACTS:
        raise HTTPException(404, 'Unknown forecast product')
    from prophet.services.verification import cached_monthly_accuracy
    try:
        return await cached_monthly_accuracy(product)
    except Exception:
        logger.exception('Forecast verification read failed')
        raise HTTPException(503, 'Forecast verification is unavailable') from None
