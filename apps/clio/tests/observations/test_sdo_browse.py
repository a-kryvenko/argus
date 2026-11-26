from datetime import UTC, datetime, timedelta
import hashlib
import io
from unittest.mock import Mock

from PIL import Image
import pytest

from clio.config import load_observation_config
from clio.ingestion.extras import sdo_browse as service
from clio.ingestion.utils.file_observations import find
from clio.providers import sdo as provider

NOW = datetime(2026, 11, 2, 10, 30, tzinfo=UTC)


def jpeg(color='red', size=4096):
    output = io.BytesIO()
    Image.new('RGB', (size, size), color).save(output, 'JPEG')
    return output.getvalue()


@pytest.fixture
def browse(storage, monkeypatch):
    monkeypatch.setattr(service, 'SIZES', (128, 512, 4096))
    downloaded = Mock(return_value=(jpeg(), NOW))
    paths = []
    def download(channel, path):
        paths.append(path)
        content, updated = downloaded(channel)
        path.write_bytes(content)
        return updated
    monkeypatch.setattr(provider, 'download_browse_file', download)
    return downloaded, paths


def collect(now=NOW):
    return service.collect(load_observation_config().observations['aia_193_browse'],
        channel='aia193', source='nasa.aia_browse', mode='live', now=now)


def test_images_are_prepared_in_tmp_and_only_completed_uploads_are_saved(storage, browse):
    _, paths = browse
    assert collect()['status'] == 'complete'
    row = find('aia_193_browse')[0]
    assert len(storage.objects) == 6
    assert row['sha256'] == hashlib.sha256(jpeg()).hexdigest()
    assert len(row['details']['extra_files']) == 5
    assert all(not path.parent.exists() for path in paths)
    assert row['file']['key'].endswith('original.jpg')
    assert any(key.endswith('/512.webp') for _, key in storage.objects)
    assert row['details']['captured_at'] is None
    assert row['details']['source_updated_at'] == NOW.isoformat()


def test_unchanged_image_does_not_create_extra_database_rows(storage, browse):
    first = collect()
    repeated = collect(NOW+timedelta(minutes=5))
    assert first['received'] == 1 and first['retained'] == 0
    assert repeated['received'] == 0 and repeated['retained'] == 1
    assert len(find('aia_193_browse')) == 1 and len(storage.objects) == 6


def test_interrupted_upload_stays_hidden_and_next_attempt_completes(storage, browse):
    downloaded, paths = browse
    collect()
    downloaded.return_value = (jpeg('blue'), NOW+timedelta(hours=1))
    put = storage.client.put_object.side_effect
    def fail(**kwargs):
        if kwargs['Key'].endswith('512.webp'):
            raise OSError('upload interrupted')
        return put(**kwargs)
    storage.client.put_object.side_effect = fail
    assert collect(NOW+timedelta(hours=1))['failed'] == 1
    assert len(find('aia_193_browse')) == 1
    assert len(storage.objects) == 10
    assert all(not path.parent.exists() for path in paths)
    storage.client.put_object.side_effect = put
    assert collect(NOW+timedelta(hours=1))['status'] == 'complete'
    assert downloaded.call_count == 3 and len(find('aia_193_browse')) == 2
    assert len(storage.objects) == 12


def test_invalid_download_is_not_uploaded(storage, browse):
    downloaded, paths = browse
    downloaded.return_value = (jpeg(size=512), NOW)
    assert collect()['failed'] == 1
    assert not storage.objects and not find('aia_193_browse')
    assert all(not path.parent.exists() for path in paths)
