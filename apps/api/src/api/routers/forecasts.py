from api.services.errors import ArtifactNotReadyError
from common.schemas.forecast_verification import ForecastVerification
from common.schemas.solar_wind_events import SolarWindEventForecast
from typing import Annotated, Literal

from api.schemas.forecast import Forecast
from api.schemas.metrics import ForecastMetrics
from api.schemas.metadata import MetaQuery
from common.schemas.response import ApiResponse, success_response
from api.services.forecasts.products import (
    load_forecast,
    load_metrics,
    PRODUCTS
)
from fastapi import APIRouter, HTTPException, Path

router = APIRouter(tags=["forecasts"])

TARGET_DESCRIPTION = "Available targets: " + ", ".join(f"`{target}`" for target in PRODUCTS)
AVAILABILITY_DESCRIPTION = (
    TARGET_DESCRIPTION
    + "\n\n### Product variables\n\n"
    "| target | variable (unit) | forecast type | maximum horizon (hours) |\n"
    "| --- | --- | --- | --- |\n"
    + "\n".join(
        f"| `{target}` | `{variable.name}` ({variable.unit}) | "
        + ("Quantiles: q10, q50, q90" if variable.quantiles else
           "Probability of value >= " + ", ".join(f"{t:g}" for t in variable.thresholds))
        + f" | {product.max_horizon_hours} |"
        for target, product in PRODUCTS.items()
        for variable in product.variables
    )
    + "\n\nOnly available data is returned. If no usable data is available, "
    "the endpoint returns **503**. Examples use illustrative values."
)


def _product(target: str):
    product = PRODUCTS.get(target)
    if product is None:
        raise HTTPException(status_code=404, detail="Forecast product not found")
    return product


TargetType = Literal[*PRODUCTS.keys()]


@router.get('/forecasts/solar-wind-events',
            response_model=ApiResponse[SolarWindEventForecast],
            summary='Get CME arrivals and conditional solar-wind speed forecasts',
            description='Published DONKI/ENLIL Earth impacts with a separate CME-conditioned speed outlook. '
                        'Nominal, early and late scenarios retain the event peak. Scenario envelopes are not '
                        'calibrated quantiles or hit probabilities. HSS forecasts are disabled in the current release.')
def solar_wind_events():
    from api.services.forecasts.client import read_event_forecast
    try:
        return success_response(read_event_forecast())
    except ArtifactNotReadyError:
        raise HTTPException(status_code=503, detail='Solar-wind event forecast is not ready') from None


@router.get(
    "/forecasts/{target}",
    response_model=ApiResponse[Forecast],
    summary="Get a forecast",
    description=AVAILABILITY_DESCRIPTION,
    responses={404: {"model": ApiResponse[None], "description": "Unknown product"},
               503: {"model": ApiResponse[None], "description": "Forecast is not ready"}},
)
def forecast(
    target: Annotated[TargetType, Path(description=TARGET_DESCRIPTION)],
    meta: MetaQuery = False,
):
    try:
        return success_response(load_forecast(_product(target), meta=meta))
    except ArtifactNotReadyError:
        raise HTTPException(status_code=503, detail="Forecast is not ready")


@router.get(
    "/forecasts/{target}/metrics",
    response_model=ApiResponse[ForecastMetrics],
    summary="Get forecast metrics",
    description=AVAILABILITY_DESCRIPTION,
    responses={404: {"model": ApiResponse[None], "description": "Unknown product"},
               503: {"model": ApiResponse[None], "description": "Metrics are not ready"}},
)
def metrics(
    target: Annotated[str, Path(description=TARGET_DESCRIPTION)],
    meta: MetaQuery = False,
):
    try:
        return success_response(load_metrics(_product(target), meta=meta))
    except ArtifactNotReadyError:
        raise HTTPException(status_code=503, detail="Metrics are not ready")


@router.get('/forecasts/{target}/verification',
            response_model=ApiResponse[ForecastVerification],
            summary='Get last 30 days published forecast accuracy')
def verification(target: str):
    product = _product(target)
    from api.services.forecasts.client import read_verification
    try:
        return success_response(read_verification(product.target))
    except ArtifactNotReadyError:
        raise HTTPException(status_code=503, detail='Forecast verification is unavailable')
