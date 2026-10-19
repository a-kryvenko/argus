from contextlib import asynccontextmanager
import os
import asyncio
from contextlib import suppress
import time

import sentry_sdk
from common.config import get_config
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.services.monitoring.statistics import record, run_flush_loop, flush
from api.routers.dashboard import router as dashboard_router
from api.db.session import dispose_engine
from api.routers.forecasts import router as forecasts_router
from api.routers.atmospheric_density import router as atmospheric_density_router
from api.routers.leo_drag import router as leo_drag_router
from api.routers.drag_monitor import router as drag_monitor_router
from api.routers.density_advisory import router as density_advisory_router
from api.routers.healthcheck import router as healthcheck_router
from api.routers.observations import router as observations_router
from common.schemas.response import error_response

config = get_config()


@asynccontextmanager
async def lifespan(_: FastAPI):
    from api.services.monitoring.project import run_monitor_loop
    from api.services.monitoring.edge_traffic import run_traffic_loop
    tasks = [asyncio.create_task(loop()) for loop in (run_flush_loop, run_monitor_loop, run_traffic_loop)]
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        with suppress(TimeoutError):
            await asyncio.wait_for(flush(), timeout=5)
        await dispose_engine()

if not config.debug:
    sentry_sdk.init(
    dsn=os.getenv("SENTRY_COLLECT_POINT"),
    send_default_pii=True,
)

app = FastAPI(
    title="ARGUS SUNWATCH API",
    debug=config.debug,
    root_path="/api/v1",
    lifespan=lifespan,
)

if not config.debug:
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        sentry_sdk.capture_exception(exc)

        return JSONResponse(
            content=error_response(
                code="INTERNAL_ERROR",
                msg="Something went wrong..."
            ).model_dump(exclude_none=True),
            status_code=500,
        )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    code = {
        404: "NOT_FOUND",
        503: "NOT_READY",
    }.get(exc.status_code, "HTTP_ERROR")
    return JSONResponse(
        content=error_response(
            code=code,
            msg=exc.detail
        ).model_dump(exclude_none=True),
        status_code=exc.status_code,
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard_router, include_in_schema=False)
app.include_router(drag_monitor_router, include_in_schema=False)
app.include_router(density_advisory_router, include_in_schema=False)
app.include_router(healthcheck_router)
app.include_router(observations_router)

app.include_router(atmospheric_density_router)
app.include_router(leo_drag_router)
app.include_router(forecasts_router)

@app.middleware("http")
async def add_processing_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        if request.url.path.startswith('/dashboard') or '/dashboard/' in request.url.path:
            response.headers['Cache-Control'] = 'no-store'
        return response
    finally:
        route = request.scope.get('route')
        record(getattr(route, 'path', '__unmatched__'), request.method, status_code,
               (time.perf_counter() - start_time) * 1000,
               user_agent=request.headers.get('user-agent', ''))
