import asyncio
from contextlib import asynccontextmanager

from common.config import get_config
from common.schemas.response import error_response
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy import text

from argus_clio.db.session import dispose_engine, get_session_factory
from argus_clio.routers import (
    observations, solar_wind, geomagnetic, observation_summary, collection_status,
    forecast_inputs, browse,
)

get_config()


@asynccontextmanager
async def lifespan(_):
    from argus_clio.commands._runner import setup_sentry
    setup_sentry()
    try:
        yield
    finally:
        await dispose_engine()


app = FastAPI(title='Clio observation service', version='1', lifespan=lifespan)
for module in (observations, solar_wind, geomagnetic, observation_summary,
               collection_status, forecast_inputs, browse):
    app.include_router(module.router, dependencies=[Depends(forecast_inputs.require_service_token)])


@app.exception_handler(StarletteHTTPException)
async def http_error(_, exc):
    return JSONResponse(status_code=exc.status_code, content=error_response(
        code={404: 'NOT_FOUND', 503: 'NOT_READY'}.get(exc.status_code, 'HTTP_ERROR'),
        msg=exc.detail,
    ).model_dump(exclude_none=True))


@app.get('/health/live')
async def live():
    return {'service': 'clio', 'status': 'ok'}


@app.get('/health/ready')
async def ready():
    try:
        async with get_session_factory()() as session:
            await asyncio.wait_for(session.execute(text('SELECT 1 FROM clio.scheduled_job LIMIT 1')), timeout=5)
    except Exception:
        raise HTTPException(503, 'Clio storage is not ready') from None
    return {'service': 'clio', 'status': 'ok'}
