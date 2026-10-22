"""Read complete Prophet releases over HTTP, with no filesystem/SQL fallback."""
import io
import csv
import os

import httpx
import pandas as pd
from common.schemas.forecast_release import ForecastRelease, PRODUCT_ARTIFACTS
from api.services.errors import ArtifactNotReadyError

def read_event_forecast():
    from common.schemas.solar_wind_events import SolarWindEventForecast
    release = read_release('solar-wind-events')
    row = next(csv.DictReader(io.StringIO(release.artifacts[0].csv_text)))
    return SolarWindEventForecast.model_validate_json(row['report_json'])


def read_release(product: str) -> ForecastRelease:
    if product not in PRODUCT_ARTIFACTS:
        raise ValueError('Unsupported forecast product')
    url = os.getenv('FORECASTS_URL')
    token = os.getenv('FORECASTS_SERVICE_TOKEN')
    if not url or not token:
        raise ArtifactNotReadyError('Forecast service is not configured')
    try:
        with httpx.Client(timeout=httpx.Timeout(60, connect=10), follow_redirects=False, trust_env=False) as client:
            response = client.get(url.rstrip('/') + f'/internal/v1/forecasts/{product}/latest',
                                  headers={'Authorization': f'Bearer {token}'})
            response.raise_for_status()
        release = ForecastRelease.model_validate_json(response.content)
        if release.product != product:
            raise ValueError('Unexpected forecast product')
        return release
    except (httpx.HTTPError, ValueError):
        raise ArtifactNotReadyError('Forecast service is unavailable') from None


def read_frames(product: str) -> dict[str, pd.DataFrame]:
    release = read_release(product)
    return {artifact.name: pd.read_csv(io.StringIO(artifact.csv_text), parse_dates=['issue_time', 'valid_time'])
            for artifact in release.artifacts}


def read_verification(product: str):
    from common.schemas.forecast_verification import ForecastVerification
    url, token = os.getenv('FORECASTS_URL'), os.getenv('FORECASTS_SERVICE_TOKEN')
    if not url or not token:
        raise ArtifactNotReadyError('Forecast service is not configured')
    try:
        with httpx.Client(timeout=httpx.Timeout(20, connect=10), follow_redirects=False, trust_env=False) as client:
            response = client.get(url.rstrip('/') + f'/internal/v1/forecasts/{product}/verification',
                                  headers={'Authorization': f'Bearer {token}'})
            response.raise_for_status()
        result = ForecastVerification.model_validate(response.json())
        if result.product != product:
            raise ValueError('Unexpected forecast product')
        return result
    except (httpx.HTTPError, ValueError):
        raise ArtifactNotReadyError('Forecast verification is unavailable') from None
