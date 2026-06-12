from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

import sentry_sdk

from app.routers.auth import router as auth_router
from app.routers.public.forecast import router as public_forecast_router
from app.routers.public.metrics import router as metrics_router
from app.routers.public.observations import router as observations_router
from app.routers.private.forecast import router as private_forecast_router
from app.routers.private.probability import router as private_probability_router
from app.routers.private.risk import router as private_risk_router
from app.routers.private.model import router as private_model_router


from common.config import get_config

import os

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

app.include_router(public_forecast_router)
app.include_router(metrics_router)
app.include_router(observations_router)

app.include_router(private_forecast_router)
app.include_router(private_probability_router)
app.include_router(private_risk_router)
app.include_router(private_model_router)
