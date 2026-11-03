"""Image and movie bytes are served by object storage, never the application API."""
import httpx
import pytest

from api.main import app


@pytest.mark.anyio
@pytest.mark.parametrize('path', ['/sdo', '/sdo/images/2026103012_094_' + 'a'*64 + '_256.webp', '/sdo/movies/0131.mp4'])
async def test_file_proxy_routes_are_not_exposed(path):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://api') as client:
        assert (await client.get(path)).status_code == 404
