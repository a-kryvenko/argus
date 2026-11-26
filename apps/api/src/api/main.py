from contextlib import asynccontextmanager
from time import perf_counter

import sentry_sdk
from common.config import get_config
from common.db.session import dispose_engine
from common.runtime import setup_sentry
from common.schemas.response import error_response
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.routers.dashboard import router as dashboard_router
from api.routers.density_advisory import router as density_advisory_router
from api.routers.drag_monitor import router as drag_monitor_router
from api.routers.healthcheck import router as healthcheck_router
from api.services.authentication import dashboard_origins
from api.services.upstream import upstream_client


PREFIX = '/api/v1'


@asynccontextmanager
async def lifespan(application: FastAPI):
    setup_sentry(include_request_data=False)
    try:
        async with upstream_client() as intelligence, upstream_client(max_connections=16) as monitoring:
            application.state.intelligence_client = intelligence
            application.state.monitoring_client = monitoring
            yield
    finally:
        await dispose_engine()


def failure(status: int, code: str, message: str, headers=None):
    return JSONResponse(
        error_response(code=code, msg=message).model_dump(exclude_none=True),
        status_code=status,
        headers=headers,
    )


def create_app() -> FastAPI:
    debug = get_config().debug
    application = FastAPI(
        title='ARGUS SUNWATCH Dashboard',
        debug=debug,
        lifespan=lifespan,
        openapi_url=None,
    )
    application.state.dashboard_origins = tuple(dashboard_origins())

    @application.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        code = {404: 'NOT_FOUND', 503: 'NOT_READY'}.get(exc.status_code, 'HTTP_ERROR')
        message = exc.detail if isinstance(exc.detail, str) else 'Request failed'
        return failure(exc.status_code, code, message, exc.headers)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, __: RequestValidationError):
        return failure(422, 'INVALID_INPUT', 'Request fields are invalid')

    @application.middleware('http')
    async def response_headers(request: Request, call_next):
        start = perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            if debug:
                raise
            sentry_sdk.capture_exception(exc)
            response = failure(500, 'INTERNAL_ERROR', 'Internal server error')
        response.headers['X-Process-Time'] = str(perf_counter() - start)
        path = request.url.path
        if path == f'{PREFIX}/dashboard' or path.startswith(f'{PREFIX}/dashboard/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    application.add_middleware(
        CORSMiddleware,
        allow_origins=application.state.dashboard_origins,
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
        expose_headers=['Retry-After'],
    )

    router = APIRouter(prefix=PREFIX)
    router.include_router(dashboard_router)
    router.include_router(drag_monitor_router)
    router.include_router(density_advisory_router)
    router.include_router(healthcheck_router)
    application.include_router(router)
    return application


app = create_app()
