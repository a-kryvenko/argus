"""Public views are assembled from Clio's compact numeric resources."""
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import httpx
from common.config import get_config
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import observations
from api.services import project_status
from api.services.observations import client as transport

NOW = datetime.now(UTC).replace(second=0, microsecond=0)


def configure(monkeypatch, respond):
    get_config()
    monkeypatch.setenv('OBSERVATIONS_URL', 'http://clio')
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'secret')
    original = httpx.AsyncClient
    calls = []
    def handler(request):
        calls.append(request)
        assert request.headers['authorization'] == 'Bearer secret'
        assert request.url.path.rsplit('/', 1)[-1] in {'latest', 'history'}
        return httpx.Response(200, json={'success': True, 'data': respond(request)})
    monkeypatch.setattr(transport.httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(handler)))
    app = FastAPI()
    app.include_router(observations.router)
    return TestClient(app), calls


def test_current_values_preserve_independent_times_and_build_optional_metadata(monkeypatch):
    def respond(request):
        return {'points': [dict(time=(NOW-timedelta(hours=4)).isoformat(), kp=3.),
                           dict(time=(NOW-timedelta(minutes=2)).isoformat(), v=400., bz=-5.)]}
    client, calls = configure(monkeypatch, respond)
    response = client.get('/observations?observations=v,bz,kp&meta=true')
    assert response.status_code == 200, response.text
    data = response.json()['data']
    assert data['v']['value'] == 400 and data['kp']['value'] == 3
    assert data['v']['observed_at'] != data['kp']['interval_start']
    assert data['meta']['series']['bz']['coordinate_system'] == 'GSM'
    assert {request.url.path for request in calls} == {'/internal/v1/observations/latest'}


def test_api_selects_hourly_history_and_clio_only_returns_points(monkeypatch):
    def respond(request):
        assert request.url.params['interval'] == '1h'
        assert request.url.params['metrics'] == 'bz,v'
        return {'interval': '1h', 'points': [dict(time=(NOW-timedelta(hours=2)).isoformat(), bz=-3., v=420.)]}
    client, calls = configure(monkeypatch, respond)
    response = client.get('/observations/history', params=dict(observations='bz,v', resolution='1h',
        **{'from': (NOW-timedelta(hours=4)).isoformat(), 'to': NOW.isoformat()}))
    assert response.status_code == 200, response.text
    data = response.json()['data']
    assert data['series']['bz']['points'][0]['value'] == -3.
    assert data['resolution_seconds'] == 3600
    assert len(calls) == 1 and calls[0].url.path.endswith('/history')


def test_dashboard_status_is_derived_from_values_and_configured_freshness(monkeypatch):
    config = Mock(project_config={'clio': {'observations': {
        'bz': {'freshness_minutes': 10}, 'dst': {'freshness_minutes': 120}, 'v': {'freshness_minutes': 10}}}})
    monkeypatch.setattr(project_status, 'get_config', lambda: config)
    result = project_status.status_from_points([
        dict(time=(NOW-timedelta(minutes=11)).isoformat(), bz=-2.),
        dict(time=(NOW-timedelta(hours=2)).isoformat(), dst=-10.)], now=NOW)
    values = {item['metric']: item for item in result['measurements']}
    assert values['bz']['status'] == 'delayed'
    assert values['dst']['status'] == 'fresh' and values['dst']['age_seconds'] == 3600
    assert values['v']['status'] == 'unavailable'
    assert result['status'] == 'degraded'


def test_index_trends_read_observation_history_without_as_of(monkeypatch):
    def respond(request):
        assert 'as_of' not in request.url.params
        if request.url.path.endswith('/latest'):
            return {'points': [dict(time=NOW.isoformat(), ap=10, f10_7=120)]}
        assert request.url.params['interval'] == '1m'
        return {'interval': '1m', 'points': [
            dict(time=(NOW-timedelta(hours=3)).isoformat(), ap=None, f10_7=100),
            dict(time=(NOW-timedelta(hours=2)).isoformat(), ap=5, f10_7=None)]}
    client, calls = configure(monkeypatch, respond)
    response = client.get('/observations?observations=ap,f10_7&trends=true')
    assert response.status_code == 200, response.text
    assert response.json()['data']['changes_1h'] == {'ap': 5, 'f10_7': 20}
    assert [request.url.path.rsplit('/', 1)[-1] for request in calls] == ['latest', 'history']
