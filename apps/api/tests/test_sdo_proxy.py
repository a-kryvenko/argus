import asyncio
from unittest.mock import Mock

from fastapi import FastAPI
import httpx
import pytest

from api.routers import sdo
from api.services import sdo as service

NAME = f'2026103012_094_{"a"*64}_256.webp'
ASSET = NAME.rsplit('_', 1)[0]
AsyncClient = httpx.AsyncClient


def app():
    application = FastAPI()
    application.include_router(sdo.router)
    return application


def test_standalone_sdo_catalog_is_not_exposed():
    async def verify():
        async with AsyncClient(transport=httpx.ASGITransport(app=app()), base_url='http://api') as client:
            assert (await client.get('/sdo')).status_code == 404
    asyncio.run(verify())


@pytest.mark.parametrize('status,expected', [(200, 200), (304, 304), (404, 404), (401, 503), (500, 503)])
def test_image_proxy_authenticates_and_releases_stream_to_shared_client(monkeypatch, status, expected):
    monkeypatch.setenv('OBSERVATIONS_URL', 'http://clio')
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'secret')
    responses = []
    def upstream(request):
        assert request.url.path == '/internal/v1/observations/sdo-browse/'+NAME
        assert request.headers['authorization'] == 'Bearer secret'
        assert request.headers['if-none-match'] == '"test"'
        response = httpx.Response(status, content=b'image' if status == 200 else b'', headers={'ETag': '"test"'})
        responses.append(response)
        return response
    upstream_client = AsyncClient(transport=httpx.MockTransport(upstream))
    monkeypatch.setattr(sdo, 'image_client', lambda: upstream_client)
    async def verify():
        async with AsyncClient(transport=httpx.ASGITransport(app=app()), base_url='http://api') as client:
            response = await client.get('/sdo/images/'+NAME, headers={'If-None-Match': '"test"'})
            assert response.status_code == expected
            if status in (200, 304):
                assert 'immutable' in response.headers['cache-control']
                assert response.headers['etag'] == '"test"'
                assert response.content == (b'image' if status == 200 else b'')
            if status == 200:
                assert response.headers['content-type'] == 'image/webp'
            assert 'secret' not in response.text
        assert not upstream_client.is_closed
        assert all(response.is_closed for response in responses)
        await upstream_client.aclose()
    asyncio.run(verify())


def test_invalid_images_never_contact_clio(monkeypatch):
    monkeypatch.setattr(sdo, 'image_client', lambda: pytest.fail('Unexpected Clio read'))
    async def verify():
        async with AsyncClient(transport=httpx.ASGITransport(app=app()), base_url='http://api') as client:
            for name in ['missing.jpg', NAME.replace('094', '999'), NAME.replace('256', '250')]:
                assert (await client.get('/sdo/images/'+name)).status_code == 404
    asyncio.run(verify())


def test_image_client_pool_is_reused_and_closed_on_shutdown(monkeypatch):
    client = AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    factory = Mock(return_value=client)
    monkeypatch.setattr(service, '_client', None)
    monkeypatch.setattr(service.httpx, 'AsyncClient', factory)
    assert service.image_client() is service.image_client() is client
    factory.assert_called_once()
    asyncio.run(service.close_image_client())
    assert client.is_closed and service._client is None
