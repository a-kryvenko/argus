import asyncio
from datetime import UTC, datetime, timedelta
import io
import json
from unittest.mock import Mock

from fastapi import Depends, FastAPI
import httpx
from PIL import Image
import pytest
import requests

from clio.observations import sdo_browse as service
from clio.routers import sdo_browse as routes
from clio.routers.auth import require_service_token
from clio.config import ClioObservations
from clio.ingestion.collect import collect

NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)


def jpeg(color='orange', size=4096):
    buffer = io.BytesIO()
    Image.new('RGB', (size, size), color).save(buffer, 'JPEG')
    return buffer.getvalue()


def test_derivatives_are_real_images_and_published_only_when_complete(tmp_path, monkeypatch):
    original = jpeg()
    monkeypatch.setenv('ARGUS_SDO_BROWSE_ARCHIVE', str(tmp_path))
    download = Mock(return_value=(original, NOW-timedelta(minutes=5)))
    monkeypatch.setattr(service, 'download', download)
    record = service.snapshot(tmp_path, '094', NOW, None)
    image = record['image']
    for size in service.SIZES:
        for extension, format in [('jpg', 'JPEG'), ('webp', 'WEBP')]:
            path = service.image_path(f'{image["asset_id"]}_{size}.{extension}')
            with Image.open(path) as result:
                assert result.size == (size, size) and result.format == format
    assert service.image_path(f'{image["asset_id"]}_4096.jpg').read_bytes() == original
    assert json.loads((tmp_path / f'{image["asset_id"]}.json').read_text()) == image
    assert service.snapshot(tmp_path, '094', NOW+timedelta(minutes=20), record) == record
    download.assert_called_once()
    # A new poll of the same source must not create another image or reset its age.
    unchanged = service.snapshot(tmp_path, '094', NOW+timedelta(hours=1), record)
    assert unchanged['image'] == image
    assert len(list(tmp_path.glob('*.jpg'))) == 6


def test_two_newest_per_channel_survive_partial_failure_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv('ARGUS_SDO_BROWSE_ARCHIVE', str(tmp_path))
    monkeypatch.setattr(service, 'CHANNELS', ('094', '131'))
    monkeypatch.setattr(service, 'SIZES', (128,))
    for hour, color in enumerate(('red', 'green', 'blue')):
        original = jpeg(color)
        monkeypatch.setattr(service, 'download', lambda channel: (original, NOW+timedelta(hours=hour)))
        assert not service.collect(NOW+timedelta(hours=hour))['failed']
    assert len(service.catalog().images) == 2
    receipts = [path for path in tmp_path.glob('*.json') if path.name != 'latest.json']
    assert len(receipts) == 4
    assert all(not path.name.startswith('2026103012') for path in receipts)
    latest = (tmp_path / 'latest.json').read_bytes()
    def fail(channel):
        raise requests.Timeout('NASA unavailable')
    monkeypatch.setattr(service, 'download', fail)
    assert service.collect(NOW+timedelta(days=10))['failed'] == ['094', '131']
    assert (tmp_path / 'latest.json').read_bytes() == latest
    assert service.cleanup() == 0
    assert all(path.exists() for path in receipts)


def test_invalid_or_failed_derivatives_never_replace_previous_image(tmp_path, monkeypatch):
    monkeypatch.setenv('ARGUS_SDO_BROWSE_ARCHIVE', str(tmp_path))
    monkeypatch.setattr(service, 'CHANNELS', ('094',))
    monkeypatch.setattr(service, 'SIZES', (128,))
    monkeypatch.setattr(service, 'download', lambda _: (jpeg(), None))
    service.collect(NOW)
    previous = service.catalog()
    monkeypatch.setattr(service, 'download', lambda _: (jpeg('red', size=512), None))
    assert service.collect(NOW+timedelta(hours=1))['failed'] == ['094']
    assert service.catalog() == previous
    monkeypatch.setattr(service, 'download', lambda _: (jpeg('red'), None))
    save = service.atomic_write
    def fail_webp(path, content):
        if path.suffix == '.webp':
            raise OSError('disk failure')
        return save(path, content)
    monkeypatch.setattr(service, 'atomic_write', fail_webp)
    assert service.collect(NOW+timedelta(hours=1))['failed'] == ['094']
    assert service.catalog() == previous


def test_browse_only_collection_uses_no_numeric_or_scientific_pipeline(monkeypatch):
    run = Mock(return_value={'status': 'complete', 'failed': []})
    monkeypatch.setattr(service, 'collect', run)
    config = ClioObservations(observations={}, sdo_browse_enabled=True)
    result = asyncio.run(collect(config, ['sdo-browse'], now=NOW))
    run.assert_called_once_with(NOW)
    assert result['sdo_browse']['status'] == 'complete' and result['downloaded_measurements'] == 0


def test_clio_auth_and_immutable_images_support_conditional_get(tmp_path, monkeypatch):
    monkeypatch.setenv('ARGUS_SDO_BROWSE_ARCHIVE', str(tmp_path))
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'test-secret')
    name = f'2026103012_094_{"a"*64}_256.webp'
    (tmp_path / name).write_bytes(b'webp')
    app = FastAPI()
    app.include_router(routes.router, dependencies=[Depends(require_service_token)])
    async def verify():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://clio') as client:
            base = '/internal/v1/observations/sdo-browse'
            assert (await client.get(base)).status_code == 401
            headers = {'Authorization': 'Bearer test-secret'}
            assert (await client.get(base, headers=headers)).json()['data']['images'] == []
            response = await client.get(base+'/'+name, headers=headers)
            assert response.content == b'webp' and response.headers['content-type'] == 'image/webp'
            assert 'immutable' in response.headers['cache-control']
            headers['If-None-Match'] = response.headers['etag']
            assert (await client.get(base+'/'+name, headers=headers)).status_code == 304
            for invalid in ['missing.webp', name.replace('094', '999'), name.replace('256', '250')]:
                assert (await client.get(base+'/'+invalid, headers=headers)).status_code == 404
    asyncio.run(verify())
