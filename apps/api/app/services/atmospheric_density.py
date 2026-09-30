"""Render atmospheric density from a published Prophet release."""
from common.config import get_config
from common.density_contract import parse_density_frame
from common.schemas.atmospheric_density import DensityForecast
from app.services.forecast_products import ArtifactNotReadyError
from app.services.forecasts_client import read_frames


def load_density_forecast() -> DensityForecast:
    config = get_config()
    entry = config.models_registry["models"]["atmospheric_density"]
    try:
        frame = read_frames("atmospheric-density")["atmospheric_density"]
        return parse_density_frame(frame, max_age_hours=entry.get("max_age_hours", 6))
    except (OSError, ValueError, KeyError) as exc:
        raise ArtifactNotReadyError("Atmospheric density forecast is not ready") from exc
