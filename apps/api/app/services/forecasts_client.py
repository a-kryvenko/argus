"""Read complete Prophet releases over HTTP, with no filesystem/SQL fallback."""
import io
import os

import httpx
import pandas as pd
from common.schemas.forecast_release import ForecastRelease, PRODUCT_ARTIFACTS
from forecast.exceptions import ArtifactNotReadyError

MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def read_release(product: str) -> ForecastRelease:
    if product not in PRODUCT_ARTIFACTS:
        raise ValueError('Unsupported forecast product')
    url = os.getenv('FORECASTS_URL')
    token = os.getenv('FORECASTS_SERVICE_TOKEN')
    if not url or not token:
        raise ArtifactNotReadyError('Forecast service is not configured')
    try:
        with httpx.Client(timeout=httpx.Timeout(60, connect=10), follow_redirects=False, trust_env=False) as client:
            with client.stream('GET', url.rstrip('/') + f'/internal/v1/forecasts/{product}/latest',
                               headers={'Authorization': f'Bearer {token}'}) as response:
                response.raise_for_status()
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > MAX_RESPONSE_BYTES:
                        raise ValueError('Forecast response is too large')
        release = ForecastRelease.model_validate_json(content)
        if release.product != product:
            raise ValueError('Unexpected forecast product')
        return release
    except (httpx.HTTPError, ValueError):
        raise ArtifactNotReadyError('Forecast service is unavailable') from None


def read_frames(product: str) -> dict[str, pd.DataFrame]:
    release = read_release(product)
    return {artifact.name: pd.read_csv(io.StringIO(artifact.csv_text), parse_dates=['issue_time', 'valid_time'])
            for artifact in release.artifacts}
