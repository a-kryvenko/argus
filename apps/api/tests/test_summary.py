import asyncio
from unittest.mock import AsyncMock

from fastapi import FastAPI
import httpx
import pytest

from api.routers import summary as routes
from api.services.errors import ArtifactNotReadyError
from api.services.forecasts import client as forecasts


def test_public_summary_returns_prophet_response_unchanged(monkeypatch):
    data = {'generated_at': '2026-11-07T12:00:00Z',
            'observations': {'plasma': {'items': [{'value': 455., 'risk_level': 3,
                'image_url': 'https://media.example/v.svg'}]}}, 'cme_events': []}
    read = AsyncMock(return_value=data)
    monkeypatch.setattr(routes, 'read_summary', read)
    app = FastAPI()
    app.include_router(routes.router)
    async def verify():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://api') as client:
            response = await client.get('/summary')
            assert response.status_code == 200
            assert response.json()['data'] == data
            assert response.headers['cache-control'] == 'no-store'
        read.assert_awaited_once_with()
    asyncio.run(verify())


def test_unavailable_prophet_does_not_trigger_local_summary_generation(monkeypatch):
    monkeypatch.setattr(routes, 'read_summary', AsyncMock(side_effect=ArtifactNotReadyError('private details')))
    app = FastAPI()
    app.include_router(routes.router)
    async def verify():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://api') as client:
            response = await client.get('/summary')
            assert response.status_code == 503 and 'private details' not in response.text
    asyncio.run(verify())


@pytest.mark.parametrize('status,payload', [(200, {'observations': {}}), (503, {}), (200, [])])
def test_summary_client_only_reads_prophet(monkeypatch, status, payload):
    monkeypatch.setenv('FORECASTS_URL', 'http://prophet')
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'test-token')
    requests = []
    def handle(request):
        requests.append(request)
        assert request.url.path == '/internal/v1/summary'
        assert request.headers['Authorization'] == 'Bearer test-token'
        return httpx.Response(status, json=payload)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(forecasts.httpx, 'AsyncClient', lambda **kwargs: client)
    if status == 200 and isinstance(payload, dict):
        assert asyncio.run(forecasts.read_summary()) == payload
    else:
        with pytest.raises(ArtifactNotReadyError):
            asyncio.run(forecasts.read_summary())
    assert len(requests) == 1
