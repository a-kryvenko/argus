from typing import Annotated, Literal

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
from fastapi import APIRouter, HTTPException, Path

router = APIRouter(tags=["forecasts"])

TARGETS_BY_VISIBILITY = {
    visibility: ", ".join(
        f"`{target}`"
        for target, product in PRODUCTS.items()
        if product.visibility == visibility
    )
    for visibility in ("public", "private")
}
TARGET_DESCRIPTION = "Available targets by visibility:\n\n" + "\n".join(
    f"- **{visibility}**: {targets}"
    for visibility, targets in TARGETS_BY_VISIBILITY.items()
)
AVAILABILITY_DESCRIPTION = (
    "Choose a `target` with its matching `visibility`:\n\n"
    "| visibility | target |\n"
    "| --- | --- |\n"
    + "\n".join(
        f"| `{visibility}` | {targets} |"
        for visibility, targets in TARGETS_BY_VISIBILITY.items()
    )
    + "\n\nA target requested under the wrong visibility returns **404**."
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


def _documented_responses(model, not_ready_message: str) -> dict:
    return {
        200: {"content": {"application/json": {"examples": {
            "sample": {
                "summary": "Illustrative response (one lead time)",
                "value": {"success": True, "data": model.model_config["json_schema_extra"]["examples"][0], "error": None},
            },
        }}}},
        404: {
            "model": ApiResponse[None],
            "description": "Unknown product or target/visibility mismatch.",
            "content": {"application/json": {"example": {
                "success": False, "error": {"code": "NOT_FOUND", "message": "Forecast product not found"},
            }}},
        },
        503: {
            "model": ApiResponse[None],
            "description": not_ready_message,
            "content": {"application/json": {"example": {
                "success": False, "error": {"code": "NOT_READY", "message": not_ready_message},
            }}},
        },
    }


Visibility = Annotated[
    Literal["public", "private"], Path(description=TARGET_DESCRIPTION)
]


def _product(target: str, visibility: str):
    try:
        return get_product(target, visibility)
    except ProductNotFoundError:
        raise HTTPException(status_code=404, detail="Forecast product not found")

TargetType = Literal[*PRODUCTS.keys()]

@router.get(
    "/{visibility}/forecasts/{target}",
    response_model=ApiResponse[Forecast],
    summary="Get a forecast",
    description=AVAILABILITY_DESCRIPTION,
    responses=_documented_responses(Forecast, "Forecast is not ready"),
)
def forecast(
    visibility: Visibility,
    target: Annotated[TargetType, Path(description=TARGET_DESCRIPTION)],
):
    try:
        return success_response(load_forecast(_product(target, visibility)))
    except ArtifactNotReadyError:
        raise HTTPException(status_code=503, detail="Forecast is not ready")


@router.get(
    "/{visibility}/forecasts/{target}/metrics",
    response_model=ApiResponse[ForecastMetrics],
    summary="Get forecast metrics",
    description=AVAILABILITY_DESCRIPTION,
    responses=_documented_responses(ForecastMetrics, "Metrics are not ready"),
)
def metrics(
    visibility: Visibility,
    target: Annotated[str, Path(description=TARGET_DESCRIPTION)],
):
    try:
        return success_response(load_metrics(_product(target, visibility)))
    except ArtifactNotReadyError:
        raise HTTPException(status_code=503, detail="Metrics are not ready")
