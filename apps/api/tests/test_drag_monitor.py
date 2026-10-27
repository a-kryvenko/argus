from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from api.services.authentication import current_user
from api.db.session import get_request_db_session
from api.routers import drag_monitor


def test_session_owner_and_write_origin(monkeypatch):
    app = FastAPI()
    app.include_router(drag_monitor.router)
    async def no_database():
        yield None
    app.dependency_overrides[get_request_db_session] = no_database
    calls = []
    monkeypatch.setattr(drag_monitor, 'request', lambda *args, **kwargs: calls.append((args, kwargs)) or [])
    with TestClient(app) as client:
        assert client.get('/dashboard/drag-monitor').status_code == 401
        app.dependency_overrides[current_user] = lambda: SimpleNamespace(id=7)
        assert client.get('/dashboard/drag-monitor').status_code == 200
        assert calls[-1][0][:3] == ('GET', 7, '')
        profile_id = uuid4()
        assert client.post(f'/dashboard/drag-monitor/{profile_id}/assess').status_code == 403
        assert client.post(f'/dashboard/drag-monitor/{profile_id}/assess', headers={'Origin': 'http://localhost:3000'}).status_code == 200
        assert calls[-1][0][1] == 7
        assert client.get('/dashboard/drag-monitor?owner_id=999').status_code == 200
        assert calls[-1][0][1] == 7
        run_id = uuid4()
        assert client.get(f'/dashboard/drag-monitor/{profile_id}/chart?run_id={run_id}').status_code == 200
        assert calls[-1][0][1:3] == (7, f'/{profile_id}/chart')
        assert calls[-1][1]['params'] == {'run_id': str(run_id)}
        assert client.get(f'/dashboard/drag-monitor/{profile_id}/chart?run_id=invalid').status_code == 422
        assert client.get('/dashboard/drag-monitor/events?unacknowledged=true').status_code == 200
        assert calls[-1][0][1:3] == (7, '/events')
        assert calls[-1][1]['params']['unacknowledged'] == 'true'
        event_id = uuid4()
        assert client.post(f'/dashboard/drag-monitor/events/{event_id}/ack').status_code == 403
        assert client.post(f'/dashboard/drag-monitor/events/{event_id}/ack', headers={'Origin': 'http://localhost:3000'}).status_code == 200
        assert calls[-1][0][1:3] == (7, f'/events/{event_id}/ack')
        assert client.get(f'/dashboard/drag-monitor/{profile_id}/runs/{run_id}').status_code == 200
        assert calls[-1][0][1:3] == (7, f'/{profile_id}/runs/{run_id}')


@pytest.mark.parametrize(('upstream', 'expected'), [(401, 503), (404, 404), (409, 409), (422, 503), (500, 503)])
def test_transport_sanitizes_errors(monkeypatch, upstream, expected):
    monkeypatch.setenv('INTELLIGENCE_URL', 'http://internal')
    monkeypatch.setenv('INTELLIGENCE_SERVICE_TOKEN', 'test-token')
    def respond(request):
        assert request.url.path == '/internal/v1/drag-monitor/7'
        assert request.headers['Authorization'] == 'Bearer test-token'
        return httpx.Response(upstream, json={'detail': 'secret upstream content'})
    with pytest.raises(HTTPException) as raised:
        drag_monitor.request('GET', 7, '', list, transport=httpx.MockTransport(respond))
    assert raised.value.status_code == expected
    assert 'secret' not in raised.value.detail


def test_transport_contract_check(monkeypatch):
    monkeypatch.setenv('INTELLIGENCE_URL', 'http://internal')
    monkeypatch.setenv('INTELLIGENCE_SERVICE_TOKEN', 'test-token')
    with pytest.raises(HTTPException):
        drag_monitor.request('GET', 7, '', list[int], transport=httpx.MockTransport(lambda _: httpx.Response(200, json=['bad'])))
