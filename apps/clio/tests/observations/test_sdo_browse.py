from common.data import datamanager
from datetime import UTC, datetime, timedelta
import asyncio
import io
from unittest.mock import Mock
from PIL import Image
import pytest

from clio.ingestion.extras import sdo_browse as service

from clio.ingestion.utils import file_observations
from clio.providers import sdo as provider
from clio.config import load_observation_config
from clio.routers import sdo_browse as routes

NOW = datetime(2026, 11, 2, 10, 30, tzinfo=UTC)


def collect(now):
    return service.collect(load_observation_config().observations['aia_193_browse'],
        channel='aia193', source='nasa.aia_browse', mode='live', now=now)


def jpeg(color='red', size=4096):
    output = io.BytesIO()
    Image.new('RGB', (size,size), color).save(output, 'JPEG')
    return output.getvalue()


@pytest.fixture
def browse(storage, monkeypatch):
    monkeypatch.setattr(service, 'SIZES', (128, 512, 4096))
    monkeypatch.setattr(routes, 'CHANNELS', ('193',))
    monkeypatch.setattr(routes, 'SIZES', (128, 512, 4096))
    download = Mock(return_value=(jpeg(), NOW))
    monkeypatch.setattr(provider, 'download_browse', download)
    return storage, download


def test_browse_publishes_original_and_variants_without_local_archive(browse):
    storage, download = browse
    assert not collect(NOW)['failed']
    assert len(storage.objects) == 6
    for (bucket,key),(content,_,_) in storage.objects.items():
        assert bucket == 'argus-public' and key.startswith('images/AIA/193/202611021030/')
        with Image.open(io.BytesIO(content)) as image:
            size = 4096 if key.endswith('original.jpg') else int(key.rsplit('/',1)[1].split('.')[0])
            assert image.size == (size,size)
    assert not collect(NOW)['failed']
    assert download.call_count == 2
    assert not list(storage.root.glob('**/*.jpg'))


def test_catalog_uses_db_metadata_without_reading_images(browse, monkeypatch):
    storage, _ = browse
    collect(NOW)
    fail = Mock(side_effect=AssertionError('No S3 reads in catalogs'))
    monkeypatch.setattr(datamanager, 'read_reference', fail)
    monkeypatch.setattr(storage.client, 'get_object', fail)
    image = routes._catalog().images[0]
    assert image.urls['webp.512'] == 'https://media.example.test/images/AIA/193/202611021030/512.webp'
    fail.assert_not_called()


def test_generic_catalog_returns_only_completed_originals(browse, metadata_session):
    from clio.routers.observation_files import load_observation_files
    storage, _ = browse
    collect(NOW)
    files = asyncio.run(load_observation_files(metadata_session, observation='aia_193_browse',
        start=NOW-timedelta(minutes=1), end=NOW+timedelta(minutes=1)))
    assert len(files) == 1
    assert files[0].file.key.endswith('/original.jpg')
    import hashlib
    assert files[0].sha256 == hashlib.sha256(storage.objects['argus-public', files[0].file.key][0]).hexdigest()
    row = file_observations.find(kind='sdo_browse', variable='aia193', entity='jpg.original')[0]
    file_observations.set_details('sdo_browse', 'aia193', NOW, {**row['details'], 'complete': False})
    assert asyncio.run(load_observation_files(metadata_session, observation='aia_193_browse',
        start=NOW-timedelta(minutes=1), end=NOW+timedelta(minutes=1))) == []


def test_unchanged_image_updates_check_time_without_new_objects(browse):
    storage, download = browse
    collect(NOW)
    first = routes._catalog().images[0]
    collect(NOW+timedelta(hours=1))
    collect(NOW+timedelta(hours=1,minutes=10))
    assert download.call_count == 3 and len(storage.objects) == 6
    assert routes._catalog().images[0] == first


def test_interrupted_batch_keeps_previous_catalog_and_resumes_in_memory(browse):
    storage, download = browse
    collect(NOW)
    first = routes._catalog().images[0]
    download.return_value = (jpeg('blue'), NOW+timedelta(hours=1))
    def put(**kwargs):
        if '202611021130' in kwargs['Key'] and kwargs['Key'].endswith('512.webp'):
            raise OSError('upload interrupted')
        return storage.put(**kwargs)
    storage.client.put_object.side_effect = put
    assert collect(NOW+timedelta(hours=1))['failed'] == 1
    assert routes._catalog().images[0] == first
    storage.client.put_object.side_effect = storage.put
    assert not collect(NOW+timedelta(hours=1,minutes=1))['failed']
    assert routes._catalog().images[0].asset_id != first.asset_id
    assert download.call_count == 2


def test_invalid_image_does_not_publish(browse):
    storage, download = browse
    download.return_value = (jpeg(size=512), NOW)
    assert collect(NOW)['failed'] == 1
    assert not storage.objects and routes._catalog().images == []


@pytest.mark.parametrize('changes', [{'captured_at': None}, {'channel': '171'}])
def test_catalog_rejects_corrupt_completed_receipt(browse, changes, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    collect(NOW)
    monkeypatch.setattr(routes, 'CHANNELS', ('193', '171'))
    row = file_observations.find(kind='sdo_browse', variable='aia193', entity='jpg.original')[0]
    file_observations.set_details('sdo_browse', 'aia193', NOW, {**row['details'], **changes})
    app = FastAPI()
    app.include_router(routes.router)
    assert TestClient(app).get('/internal/v1/observations/sdo').status_code == 503
