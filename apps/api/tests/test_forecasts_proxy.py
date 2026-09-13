import hashlib
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest

from common.schemas.forecast_release import ForecastRelease, ForecastArtifact
from app.services import forecasts_client, forecast_products
from forecast.exceptions import ArtifactNotReadyError


def payload():
    content = 'issue_time,valid_time,lead_hours,dst_q10,dst_q50,dst_q90\n2026-09-13T00:00:00Z,2026-09-13T01:00:00Z,1,-20,-10,0\n'
    return ForecastRelease(release_id=uuid4(), run_id=uuid4(), product='dst',
        issue_time=datetime(2026, 9, 13, tzinfo=UTC), published_at=datetime.now(UTC),
        artifacts=[ForecastArtifact(name='dst_quantile', csv_text=content,
            sha256=hashlib.sha256(content.encode()).hexdigest(), row_count=1,
            columns=content.splitlines()[0].split(','), model_info={})]).model_dump(mode='json')


def install_transport(monkeypatch, response):
    monkeypatch.setenv('FORECASTS_URL', 'http://prophet-api:8000')
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'secret')
    real_client = httpx.Client
    def handle(request):
        assert request.url.path == '/internal/v1/forecasts/dst/latest'
        assert request.headers['Authorization'] == 'Bearer secret'
        return response
    monkeypatch.setattr(forecasts_client.httpx, 'Client', lambda **kwargs: real_client(
        transport=httpx.MockTransport(handle), **kwargs))


def test_public_forecast_renders_http_release_without_local_files(monkeypatch):
    install_transport(monkeypatch, httpx.Response(200, json=payload()))
    monkeypatch.setattr(forecast_products, 'get_config', lambda: pytest.fail('Forecast must not read local paths'))
    result = forecast_products.load_forecast(forecast_products.get_product('dst', 'public'))
    assert result.predictions[0].variables['dst'].continuous.q50 == -10


@pytest.mark.parametrize('failure', ['unavailable', 'corrupt', 'wrong_product', 'version', 'incomplete'])
def test_upstream_failures_do_not_fall_back_to_csv(monkeypatch, failure):
    data = payload()
    if failure == 'corrupt':
        data['artifacts'][0]['csv_text'] += 'modified'
    elif failure == 'wrong_product':
        data['product'] = 'hmf'
    elif failure == 'version':
        data['contract_version'] = 2
    elif failure == 'incomplete':
        data['artifacts'] = []
    install_transport(monkeypatch, httpx.Response(503 if failure == 'unavailable' else 200, json=data))
    with pytest.raises(ArtifactNotReadyError):
        forecasts_client.read_frames('dst')


def test_missing_token_fails_without_http(monkeypatch):
    monkeypatch.delenv('FORECASTS_SERVICE_TOKEN', raising=False)
    with pytest.raises(ArtifactNotReadyError, match='not configured'):
        forecasts_client.read_release('dst')
