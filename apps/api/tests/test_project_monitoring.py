import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi import Response
from api.services import project_status as monitoring
from api.routers.dashboard import project_monitoring

NOW = datetime(2026, 9, 17, 12, tzinfo=UTC)


def forecast_payload(product='dst', **changes):
    identifier = '00000000-0000-4000-8000-000000000001'
    stamp = NOW.isoformat()
    return {'product': product, 'assessed_at': stamp,
            'current_release': {'release_id': identifier, 'run_id': identifier, 'issue_time': stamp,
                                'published_at': stamp, 'started_at': stamp},
            'release_age_hours': 0, 'existing_public_max_age_hours': 12,
            'freshness': 'within_age_limit', 'latest_attempt': None,
            'latest_attempt_artifacts': [], **changes}


@pytest.mark.parametrize('forecast_unavailable', [False, True])
def test_each_dashboard_request_reads_current_service_status(monkeypatch, forecast_unavailable):
    policies = monitoring.get_config().project_config['clio']['observations']
    monkeypatch.setenv('OBSERVATIONS_URL', 'http://clio')
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'observation-secret')
    monkeypatch.setenv('FORECASTS_URL', 'http://prophet')
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'forecast-secret')
    requests = []
    def respond(request):
        requests.append((str(request.url), request.headers['Authorization']))
        if request.url.host == 'clio':
            stamp = datetime.now(UTC).isoformat()
            return httpx.Response(200, json={'success': True, 'data': {
                metric: {'observed_at': stamp, 'received_at': stamp, 'usable': True}
                for metric in policies}})
        return httpx.Response(503 if forecast_unavailable else 200,
            json={'products': [forecast_payload(p) for p in monitoring.PRODUCT_ARTIFACTS]})
    response = Response()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return (await project_monitoring(response, client)).data, (await project_monitoring(Response(), client)).data

    result, second = asyncio.run(run())
    assert response.headers['Cache-Control'] == 'no-store'
    assert result['status'] == ('degraded' if forecast_unavailable else 'ok')
    assert result['observations']['status'] == 'ok'
    assert {item['status'] for item in result['forecasts']} == {'unknown' if forecast_unavailable else 'ok'}
    assert set(result) == {'status', 'observations', 'forecasts', 'checked_at', 'stale'}
    assert {item['product'] for item in result['forecasts']} == set(monitoring.PRODUCT_ARTIFACTS)
    expected = [
        ('http://clio/internal/v1/observations/status', 'Bearer observation-secret'),
        ('http://prophet/internal/v1/forecasts/status', 'Bearer forecast-secret')]
    assert sorted(requests[:2]) == sorted(requests[2:]) == expected
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


def test_malformed_observation_only_marks_affected_metric_unavailable(monkeypatch):
    config = {'clio': {'observations': {metric: {'freshness_minutes': 10} for metric in ('v', 'n', 't')}}}
    monkeypatch.setattr(monitoring, 'get_config', lambda: SimpleNamespace(project_config=config))
    monkeypatch.setenv('OBSERVATIONS_URL', 'http://clio')
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'test-token')
    stamp = datetime.now(UTC)
    payload = {'success': True, 'data': {
        'v': {'observed_at': stamp.isoformat(), 'received_at': 'unknown', 'usable': True},
        'n': {'observed_at': stamp.replace(tzinfo=None).isoformat(), 'usable': True},
        't': {'observed_at': {'bad': True}, 'usable': True},
    }}

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as client:
            return await monitoring.observation_status(client)

    status = asyncio.run(run())
    assert status['status'] == 'degraded'
    assert {point['metric']: point['status'] for point in status['measurements']} == {
        'v': 'fresh', 'n': 'unavailable', 't': 'unavailable',
    }
    assert status['measurements'][0]['received_at'] is None


def test_forecast_status_keeps_other_products_when_partial(monkeypatch):
    monkeypatch.setenv('FORECASTS_URL', 'http://prophet')
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'test-token')
    payload = {'products': [
        forecast_payload('dst', note='internal diagnostic detail'),
        forecast_payload('geomagnetic-activity', current_release=None),
        forecast_payload('solar-wind-speed', assessed_at='not a timestamp'),
    ]}

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as client:
            return await monitoring.forecast_status(client)

    statuses = {item['product']: item for item in asyncio.run(run())}
    assert statuses['dst']['status'] == 'ok'
    assert 'internal diagnostic detail' not in statuses['dst']['note']
    assert statuses['geomagnetic-activity']['status'] == 'warning'
    assert statuses['solar-wind-speed']['status'] == 'unknown'
    assert statuses['atmospheric-density']['status'] == 'unknown'
