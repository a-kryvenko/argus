import os

import sentry_sdk
from common.config import get_config
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.routers.auth import router as auth_router
from app.routers.forecasts import router as forecasts_router
from app.routers.healthcheck import router as healthcheck_router
from app.routers.private.model import router as private_model_router
from app.routers.private.risk import router as private_risk_router
from app.routers.public.observations import router as observations_router
from app.schemas.response import error_response

config = get_config()

if not config.debug:
    sentry_sdk.init(
    dsn=os.getenv("SENTRY_COLLECT_POINT"),
    send_default_pii=True,
)

app = FastAPI(
    title="ARGUS SUNWATCH Public API",
    debug=config.debug,
    root_path="/api/v1"
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

# app.include_router(auth_router)
app.include_router(healthcheck_router)

app.include_router(forecasts_router)
app.include_router(observations_router)

app.include_router(private_risk_router)
app.include_router(private_model_router)
