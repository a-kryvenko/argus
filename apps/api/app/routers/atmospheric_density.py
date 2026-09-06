from fastapi import APIRouter, HTTPException

from app.schemas.atmospheric_density import DensityForecast
from app.schemas.response import ApiResponse, success_response
from app.services.atmospheric_density import load_density_forecast
from app.services.forecast_products import ArtifactNotReadyError

router = APIRouter(tags=["forecasts"])


@router.get(
    "/public/forecasts/atmospheric-density",
    response_model=ApiResponse[DensityForecast],
    summary="Get the latest atmospheric density forecast",
    description=(
        "Returns the application's precomputed 48-hour JB2008 grid from internal "
        "observations. No request body or model inputs are required. Drivers are "
        "held constant at issue time. Backgrounds use trailing 81-day means; "
        "DTC is a causal estimate from observed Dst/ap, identified in the response. "
        "Density is averaged over longitude; the "
        "longitude percentiles describe spatial variation, not uncertainty. "
        "Returns 503 when the artifact is missing, invalid or stale."
    ),
    responses={503: {"model": ApiResponse[None], "description": "Forecast is not ready"}},
)
def atmospheric_density():
    try:
        return success_response(load_density_forecast())
    except ArtifactNotReadyError:
        raise HTTPException(status_code=503, detail="Atmospheric density forecast is not ready")
