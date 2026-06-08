from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

import sentry_sdk

from app.routers.auth import router as auth_router
from app.routers.forecast import router as forecast_router
#from app.routers.metrics import router as metrics_router

from common.config import get_config

import os

config = get_config()

sentry_sdk.init(
    dsn=os.getenv("SENTRY_COLLECT_POINT"),
    send_default_pii=True,
)

app = FastAPI(
    title="ARGUS SUNWATCH Public API",
    debug=config.debug,
    root_path="/api"
)

if not config.debug:
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        sentry_sdk.capture_exception(exc)

        return JSONResponse(
            content={
                "status": "error",
                "error": "Something went wrong..."
            },
            status_code=500,
        )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        content={
            "status": "error",
            "error": exc.detail
        },
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

app.include_router(forecast_router)
# app.include_router(metrics_router)

