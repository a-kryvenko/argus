from fastapi import APIRouter, HTTPException

from apps.api.tests.reference_public.schemas.atmospheric_density import DensityForecast
from common.schemas.response import ApiResponse, success_response
from apps.api.tests.reference_public.services.forecasts.products import load_density_forecast
from apps.api.tests.reference_public.services.errors import ArtifactNotReadyError

router = APIRouter(tags=["forecasts"])


@router.get(
    "/forecasts/atmospheric-density",
    response_model=ApiResponse[DensityForecast],
    summary="Get the latest atmospheric density forecast",
    description=(
        "Returns the application's precomputed 48-hour thermospheric grid from internal "
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
        return success_response(DensityForecast.from_internal(load_density_forecast()))
    except ArtifactNotReadyError:
        raise HTTPException(status_code=503, detail="Atmospheric density forecast is not ready")
