from datetime import UTC, datetime, timedelta
import io
from unittest.mock import Mock
from PIL import Image
import pytest

from clio.observations import sdo_browse as service, file_storage, file_observations
from clio.providers import sdo as provider

NOW = datetime(2026, 11, 2, 10, 30, tzinfo=UTC)


def jpeg(color='red', size=4096):
    output = io.BytesIO()
    Image.new('RGB', (size,size), color).save(output, 'JPEG')
    return output.getvalue()


@pytest.fixture
def browse(storage, monkeypatch):
    monkeypatch.setattr(service, 'CHANNELS', ('193',))
    monkeypatch.setattr(service, 'SIZES', (128, 512, 4096))
    download = Mock(return_value=(jpeg(), NOW))
    monkeypatch.setattr(provider, 'download_browse', download)
    return storage, download


def test_browse_publishes_original_and_variants_without_local_archive(browse):
    storage, download = browse
    assert not service.collect(NOW)['failed']
    assert len(storage.objects) == 6
    for (bucket,key),(content,_,_) in storage.objects.items():
        assert bucket == 'argus-public' and key.startswith('images/AIA/193/202611021030/')
        with Image.open(io.BytesIO(content)) as image:
            size = 4096 if key.endswith('original.jpg') else int(key.rsplit('/',1)[1].split('.')[0])
            assert image.size == (size,size)
    assert not service.collect(NOW)['failed']
    assert download.call_count == 1
    assert not list(storage.root.glob('**/*.jpg'))


def test_catalog_uses_db_metadata_without_reading_images(browse, monkeypatch):
    storage, _ = browse
    service.collect(NOW)
    fail = Mock(side_effect=AssertionError('No S3 reads in catalogs'))
    monkeypatch.setattr(file_storage, 'read', fail)
    monkeypatch.setattr(storage.client, 'get_object', fail)
    image = service.catalog().images[0]
    assert image.urls['webp.512'] == 'https://media.example.test/images/AIA/193/202611021030/512.webp'
    fail.assert_not_called()


def test_unchanged_image_updates_check_time_without_new_objects(browse):
    storage, download = browse
    service.collect(NOW)
    first = service.catalog().images[0]
    service.collect(NOW+timedelta(hours=1))
    service.collect(NOW+timedelta(hours=1,minutes=10))
    assert download.call_count == 2 and len(storage.objects) == 6
    assert service.catalog().images[0] == first


def test_interrupted_batch_keeps_previous_catalog_and_resumes_in_memory(browse):
    storage, download = browse
    service.collect(NOW)
    first = service.catalog().images[0]
    download.return_value = (jpeg('blue'), NOW+timedelta(hours=1))
    def put(**kwargs):
        if '202611021130' in kwargs['Key'] and kwargs['Key'].endswith('512.webp'):
            raise OSError('upload interrupted')
        return storage.put(**kwargs)
    storage.client.put_object.side_effect = put
    assert service.collect(NOW+timedelta(hours=1))['failed'] == ['193']
    assert service.catalog().images[0] == first
    storage.client.put_object.side_effect = storage.put
    assert not service.collect(NOW+timedelta(hours=1,minutes=1))['failed']
    assert service.catalog().images[0].asset_id != first.asset_id
    assert download.call_count == 2


def test_invalid_image_does_not_publish(browse):
    storage, download = browse
    download.return_value = (jpeg(size=512), NOW)
    assert service.collect(NOW)['failed'] == ['193']
    assert not storage.objects and service.catalog().images == []
