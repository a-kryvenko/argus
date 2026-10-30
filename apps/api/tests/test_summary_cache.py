"""Cache behavior on an explicitly supplied disposable Redis."""
import asyncio
import json
import os
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from fastapi import HTTPException
import pytest
from redis.asyncio import Redis
from redis.exceptions import ConnectionError

from api.schemas.summary import SummarySettings
from api.services import summary


@pytest.fixture
def run_cache(monkeypatch):
    url = os.getenv('TEST_REDIS_URL')
    if not url:
        pytest.skip('Disposable TEST_REDIS_URL required')
    prefix = 'test:summary:'+uuid4().hex+':'
    monkeypatch.setattr(summary, 'SUMMARY_KEY', prefix+'json')
    monkeypatch.setattr(summary, 'IMAGE_KEY', prefix+'image:')
    monkeypatch.setattr(summary, 'settings', lambda: SummarySettings(ttl_seconds=1, image_ttl_seconds=4))
    def run(callback):
        async def scenario():
            async with Redis.from_url(url) as cache:
                monkeypatch.setattr(summary, 'get_cache', lambda: cache)
                try:
                    await callback(cache)
                finally:
                    keys = [key async for key in cache.scan_iter(match=prefix+'*')]
                    if keys:
                        await cache.delete(*keys)
        asyncio.run(scenario())
    return run


def test_cache_hits_skip_building_and_old_images_outlive_json(run_cache, monkeypatch):
    name = 'a'*64+'.svg'
    build = AsyncMock(return_value=({'image_name': name}, {name: b'<svg/>'}))
    monkeypatch.setattr(summary, 'build_summary', build)
    async def verify(cache):
        first = await summary.cached_summary()
        expiry = await cache.pttl(summary.SUMMARY_KEY)
        assert 0 < expiry <= 1000
        assert json.loads(first)['data']['image_name'] == name
        assert await summary.cached_summary() == first
        assert await cache.pttl(summary.SUMMARY_KEY) <= expiry
        build.assert_awaited_once()
        content, ttl = await summary.read_image(name)
        assert content == b'<svg/>' and 0 < ttl <= 4
        await asyncio.sleep(1.1)
        assert await cache.get(summary.SUMMARY_KEY) is None
        assert (await summary.read_image(name))[0] == b'<svg/>'
        await summary.cached_summary()
        assert build.await_count == 2
        await cache.delete(summary.IMAGE_KEY+name)
        with pytest.raises(HTTPException) as error:
            await summary.read_image(name)
        assert error.value.status_code == 404
        assert build.await_count == 2  # Image reads never rebuild a summary.
    run_cache(verify)


def test_concurrent_misses_build_one_snapshot_and_publish_all_images(run_cache, monkeypatch):
    names = ['b'*64+'.svg', 'c'*64+'.svg']
    async def build(_):
        await asyncio.sleep(.05)
        return {'images': names}, {name: b'<svg/>' for name in names}
    build = AsyncMock(side_effect=build)
    monkeypatch.setattr(summary, 'build_summary', build)
    async def verify(cache):
        responses = await asyncio.gather(*(summary.cached_summary() for _ in range(6)))
        assert len(set(responses)) == 1
        build.assert_awaited_once()
        assert json.loads(responses[0])['data']['images'] == names
        assert all([await cache.exists(summary.IMAGE_KEY+name) for name in names])
    run_cache(verify)


def test_redis_failure_is_reported_without_credential_details(monkeypatch):
    cache = Mock(get=AsyncMock(side_effect=ConnectionError('redis://secret:password@host')))
    monkeypatch.setattr(summary, 'get_cache', lambda: cache)
    with pytest.raises(HTTPException) as error:
        asyncio.run(summary.cached_summary())
    assert error.value.status_code == 503
    assert 'password' not in error.value.detail
