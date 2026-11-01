import asyncio
from datetime import UTC, datetime

import httpx
import pytest
from fastapi import Response
from api.services import project_status as monitoring
from api.routers.dashboard import project_monitoring

NOW = datetime(2026, 9, 17, 12, tzinfo=UTC)


def forecast_payload(product='dst', **changes):
    return {'product': product, 'assessed_at': NOW.isoformat(), 'current_release': None,
            'release_age_hours': None, 'existing_public_max_age_hours': None,
            'freshness': 'within_age_limit', 'latest_attempt': None,
            'latest_attempt_artifacts': [], **changes}


@pytest.mark.parametrize('forecast_unavailable', [False, True])
def test_each_dashboard_request_reads_current_service_status(monkeypatch, forecast_unavailable):
    monkeypatch.setenv('OBSERVATIONS_URL', 'http://clio')
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'observation-secret')
    monkeypatch.setenv('FORECASTS_URL', 'http://prophet')
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'forecast-secret')
    requests = []
    def respond(request):
        requests.append((str(request.url), request.headers['Authorization']))
        if request.url.host == 'clio':
            return httpx.Response(200, json={'success': True, 'data': {
                'points': [dict(time=datetime.now(UTC).isoformat(), **{m: 2. for m in monitoring.get_config().project_config['clio']['observations'] if m not in ('goes', 'gong')})]}})
        return httpx.Response(503 if forecast_unavailable else 200,
            json={'products': [forecast_payload(p) for p in monitoring.PRODUCT_ARTIFACTS]})
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: client(
        transport=httpx.MockTransport(respond), **kwargs))
    response = Response()
    result = asyncio.run(project_monitoring(response)).data
    assert response.headers['Cache-Control'] == 'no-store'
    assert result['status'] == ('degraded' if forecast_unavailable else 'ok')
    assert result['observations']['status'] == 'ok'
    assert {item['status'] for item in result['forecasts']} == {'unknown' if forecast_unavailable else 'ok'}
    assert set(result) == {'status', 'observations', 'forecasts', 'checked_at', 'stale'}
    assert {item['product'] for item in result['forecasts']} == set(monitoring.PRODUCT_ARTIFACTS)
    assert sorted(requests) == [
        ('http://clio/internal/v1/observations/latest', 'Bearer observation-secret'),
        ('http://prophet/internal/v1/forecasts/status', 'Bearer forecast-secret')]
    second = asyncio.run(project_monitoring(Response())).data
    assert len(requests) == 4
    assert second['checked_at'] >= result['checked_at']
    assert second['stale'] is False



def test_forecast_keeps_release_when_latest_generation_failed(monkeypatch):
    monkeypatch.setenv('FORECASTS_URL', 'http://prophet')
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'test')
    identifier = '00000000-0000-4000-8000-000000000001'
    payload = {'product': 'dst', 'assessed_at': NOW.isoformat(),
        'current_release': {'release_id': identifier, 'run_id': identifier, 'issue_time': NOW.isoformat(),
                            'published_at': NOW.isoformat(), 'started_at': NOW.isoformat()},
        'release_age_hours': 0, 'existing_public_max_age_hours': 24, 'freshness': 'within_age_limit',
        'latest_attempt': {'run_id': identifier, 'status': 'failed', 'started_at': NOW.isoformat(),
                           'error': 'private database credentials'}, 'latest_attempt_artifacts': []}
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={'products': [
            payload if p == 'dst' else forecast_payload(p) for p in monitoring.PRODUCT_ARTIFACTS]}))) as client:
            return await monitoring.forecast_status(client)
    result = next(item for item in asyncio.run(run()) if item['product'] == 'dst')
    assert result['status'] == 'warning'
    assert result['current_release']['published_at'] == NOW.isoformat().replace('+00:00', 'Z')
    assert 'private' not in result['latest_attempt']['error']
