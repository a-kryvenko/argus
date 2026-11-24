from contextlib import asynccontextmanager
import time

import sentry_sdk
from common.config import get_config
from common.runtime import setup_sentry
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.routers.dashboard import router as dashboard_router
from common.db.session import dispose_engine
from api.routers.drag_monitor import router as drag_monitor_router
from api.routers.density_advisory import router as density_advisory_router
from api.routers.healthcheck import router as healthcheck_router
from common.schemas.response import error_response

config = get_config()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        await dispose_engine()

setup_sentry()

app = FastAPI(
    title="ARGUS SUNWATCH Dashboard",
    debug=config.debug,
    root_path="/api/v1",
    lifespan=lifespan,
    openapi_url=None,
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
@app.middleware("http")
async def add_processing_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.perf_counter() - start_time)
    if request.url.path.startswith('/dashboard') or '/dashboard/' in request.url.path:
        response.headers['Cache-Control'] = 'no-store'
    return response
