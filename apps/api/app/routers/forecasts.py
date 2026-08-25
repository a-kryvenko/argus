from typing import Literal

from app.schemas.forecast import Forecast
from app.schemas.metrics import ForecastMetrics
from app.schemas.response import ApiResponse, success_response
from app.services.forecast_products import (
    ArtifactNotReadyError,
    ProductNotFoundError,
    get_product,
    load_forecast,
    load_metrics,
    PRODUCTS
)
from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["forecasts"])


def _product(target: str, visibility: str):
    try:
        return get_product(target, visibility)
    except ProductNotFoundError:
        raise HTTPException(status_code=404, detail="Forecast product not found")

TargetType = Literal[*PRODUCTS.keys()]

@router.get("/{visibility}/forecasts/{target}", response_model=ApiResponse[Forecast])
def forecast(visibility: Literal["public", "private"], target: TargetType):
    try:
        return success_response(load_forecast(_product(target, visibility)))
    except ArtifactNotReadyError:
        raise HTTPException(status_code=503, detail="Forecast is not ready")


@router.get(
    "/{visibility}/forecasts/{target}/metrics",
    response_model=ApiResponse[ForecastMetrics],
)
def metrics(visibility: Literal["public", "private"], target: str):
    try:
        return success_response(load_metrics(_product(target, visibility)))
    except ArtifactNotReadyError:
        raise HTTPException(status_code=503, detail="Metrics are not ready")
