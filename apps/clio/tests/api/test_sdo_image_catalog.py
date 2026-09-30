from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from clio.routers import sdo_images as routes
from common.sdo_images import image_path, save_image


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'test-token')
    monkeypatch.setattr(routes, 'archive_root', lambda: tmp_path)
    slot = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    now = slot + timedelta(minutes=30)
    metadata = dict(observed_at=slot.isoformat(), available_at=(slot + timedelta(minutes=20)).isoformat(),
                    source='https://example.test/aia.fits', sha256='original-fits-hash',
                    units='DN', preprocessing='test-observation-v1')
    save_image(tmp_path, slot, 'aia1600', np.full((512, 512), 3, np.float32), metadata, now=now)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        yield client, tmp_path, slot, now


def request(client, slot, now, **overrides):
    params = dict(start=slot.isoformat(), end=(slot + timedelta(hours=1)).isoformat(), as_of=now.isoformat())
    params.update(overrides)
    return client.get('/internal/v1/observations/sdo-images', params=params,
                      headers={'Authorization': 'Bearer test-token'})


def test_catalog_returns_json_paths_receipts_and_missing_channels(catalog):
    client, root, slot, now = catalog
    response = request(client, slot, now)
    assert response.status_code == 200, response.text
    assert response.headers['content-type'] == 'application/json'
    data = response.json()['data']
    assert len(data['items']) == 1 and len(data['missing']) == 8
    item = data['items'][0]
    assert item['shape'] == [512, 512] and item['dtype'] == 'float32'
    assert item['metadata']['channel'] == 'aia1600'
    assert item['metadata']['sha256'] == 'original-fits-hash'
    assert item['path'] == str(image_path(root.resolve(), slot, 'aia1600'))
    assert 'image' not in item and 'url' not in item
    # This is directly readable by the consumer on the shared filesystem.
    with np.load(item['path'], allow_pickle=False) as saved:
        assert saved['image'].shape == (512, 512)


def test_catalog_only_reads_metadata_and_filters_by_receipt_time(catalog, monkeypatch):
    client, _, slot, now = catalog
    from numpy.lib.npyio import NpzFile
    original = NpzFile.__getitem__
    def metadata_only(self, key):
        assert key == 'metadata', 'Catalog must not decompress pixel arrays'
        return original(self, key)
    monkeypatch.setattr(NpzFile, '__getitem__', metadata_only)
    data = request(client, slot, now, channel='aia1600').json()['data']
    assert len(data['items']) == 1 and not data['missing']
    data = request(client, slot, slot + timedelta(minutes=10), channel='aia1600').json()['data']
    assert not data['items'] and len(data['missing']) == 1


def test_catalog_requires_service_token(catalog, monkeypatch):
    client, _, _, _ = catalog
    for headers in ({}, {'Authorization': 'Bearer wrong'}):
        assert client.get('/internal/v1/observations/sdo-images', headers=headers).status_code == 401
    monkeypatch.delenv('OBSERVATIONS_SERVICE_TOKEN')
    assert client.get('/internal/v1/observations/sdo-images').status_code == 503


def test_catalog_validates_channel_and_bounded_aware_hourly_ranges(catalog):
    client, _, slot, now = catalog
    for params in (
        {'channel': 'hmi_v'}, {'channel': '../aia94'},
        {'start': slot.replace(tzinfo=None).isoformat()},
        {'start': (slot + timedelta(minutes=1)).isoformat()},
        {'end': slot.isoformat()},
        {'start': (slot - timedelta(hours=144)).isoformat()},
        {'end': (slot + timedelta(hours=2)).isoformat()},
        {'as_of': (datetime.now(UTC) + timedelta(days=1)).isoformat()},
    ):
        assert request(client, slot, now, **params).status_code == 422


def test_empty_catalog_does_not_create_storage_and_corruption_is_not_a_gap(catalog, monkeypatch):
    client, root, slot, now = catalog
    absent = root / 'absent'
    monkeypatch.setattr(routes, 'archive_root', lambda: absent)
    response = request(client, slot, now, channel='hmi_m')
    assert response.status_code == 200 and not response.json()['data']['items']
    assert not absent.exists()
    monkeypatch.setattr(routes, 'archive_root', lambda: root)
    image_path(root, slot, 'aia1600').write_bytes(b'broken archive')
    assert request(client, slot, now).status_code == 503


def test_default_catalog_is_limited_to_144_hourly_slots(catalog):
    client, _, _, now = catalog
    response = client.get('/internal/v1/observations/sdo-images',
                          params={'as_of': now.isoformat(), 'channel': 'aia1600'},
                          headers={'Authorization': 'Bearer test-token'})
    data = response.json()['data']
    assert len(data['items']) == 1 and len(data['missing']) == 143
