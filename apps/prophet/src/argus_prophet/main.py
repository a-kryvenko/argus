"""Internal HTTP reads of committed Prophet product releases."""
import logging
import os
import secrets
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from common.config import get_config
from common.schemas.forecast_release import ForecastRelease, PRODUCT_ARTIFACTS
from common.schemas.forecast_status import ForecastStatus
from argus_prophet.db.session import connect
from argus_prophet.services.releases.publication import read_release, ReleaseNotFound

get_config()
app = FastAPI(title='Prophet forecast service', version='1')
security = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


def require_service_token(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    expected = os.getenv('FORECASTS_SERVICE_TOKEN')
    if not expected:
        raise HTTPException(503, 'Forecast read service is not configured')
    if credentials is None or not secrets.compare_digest(credentials.credentials.encode(), expected.encode()):
        raise HTTPException(401, 'Invalid service credentials')


def load(product, release_id=None):
    if product not in PRODUCT_ARTIFACTS:
        raise HTTPException(404, 'Unknown forecast product')
    try:
        return read_release(product, release_id)
    except ReleaseNotFound:
        raise HTTPException(404 if release_id else 503, 'Forecast release is not available') from None
    except Exception:
        logger.exception('Forecast release read failed')
        raise HTTPException(503, 'Forecast storage is not ready') from None


@app.get('/internal/v1/forecasts/{product}/latest', response_model=ForecastRelease,
         dependencies=[Depends(require_service_token)])
def latest(product: str):
    return load(product)


@app.get('/internal/v1/forecasts/{product}/releases/{release_id}', response_model=ForecastRelease,
         dependencies=[Depends(require_service_token)])
def historical(product: str, release_id: UUID):
    return load(product, release_id)


@app.get('/internal/v1/forecasts/{product}/status', response_model=ForecastStatus,
         dependencies=[Depends(require_service_token)])
def status(product: str):
    if product not in PRODUCT_ARTIFACTS:
        raise HTTPException(404, 'Unknown forecast product')
    from argus_prophet.services.releases.status import product_status
    try:
        return product_status(product)
    except Exception:
        logger.exception('Forecast status read failed')
        raise HTTPException(503, 'Forecast storage is not ready') from None


@app.get('/health/live')
def live():
    return {'service': 'prophet', 'status': 'ok'}


@app.get('/health/ready')
def ready():
    if not os.getenv('FORECASTS_SERVICE_TOKEN'):
        raise HTTPException(503, 'Forecast read service is not configured')
    try:
        with connect() as conn:
            conn.execute('SET LOCAL statement_timeout=3000')
            conn.execute('SELECT release_id FROM prophet.current_forecast LIMIT 1')
    except Exception:
        raise HTTPException(503, 'Forecast storage is not ready') from None
    return {'service': 'prophet', 'status': 'ok'}
