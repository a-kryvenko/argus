from functools import lru_cache

from api_public.config import settings
from api_public.services.forecast_service import ForecastService


@lru_cache(maxsize=1)
def get_forecast_service() -> ForecastService:
    return ForecastService(
        settings.config_path,
        settings.model_path,
        settings.scaler_path,
        settings.embeddings_path,
        settings.embeddings_format,
    )
