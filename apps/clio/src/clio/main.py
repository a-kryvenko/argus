import asyncio
from contextlib import asynccontextmanager

from common.config import get_config
from common.schemas.response import error_response
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException

from clio.config import load_observation_config
from common.db.session import dispose_engine
from clio.db.session import get_db_session
from clio.routers import (
    browse,
    collection_status,
    geomagnetic,
    measurements,
    observation_files,
    observations,
    sdo_images,
    solar_wind,
    solar_wind_events,
)
from clio.routers.auth import require_service_token

get_config()
load_observation_config()


@asynccontextmanager
async def lifespan(_):
    from common.runtime import setup_sentry
    setup_sentry()
    try:
        yield
    finally:
        await dispose_engine()


app = FastAPI(title='Clio observation service', version='1', lifespan=lifespan)

for module in (
    observations,
    solar_wind,
    solar_wind_events,
    geomagnetic,
    collection_status,
    measurements,
    browse,
    sdo_images,
    observation_files
):
    app.include_router(module.router, dependencies=[Depends(require_service_token)])


@app.exception_handler(StarletteHTTPException)
async def http_error(_, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
        code={404: 'NOT_FOUND', 503: 'NOT_READY'}.get(exc.status_code, 'HTTP_ERROR'),
        msg=exc.detail,
    ).model_dump(exclude_none=True)
)


@app.get('/health/live')
async def live():
    return {'service': 'clio', 'status': 'ok'}


@app.get('/health/ready')
async def ready():
    try:
        async with get_db_session() as session:
            await asyncio.wait_for(session.execute(text('SELECT 1 FROM measurement LIMIT 1')), timeout=5)
    except Exception:
        raise HTTPException(503, 'Clio storage is not ready') from None
    return {'service': 'clio', 'status': 'ok'}
