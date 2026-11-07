from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from clio.ingestion.extras import sdo
from clio.ingestion.utils.file_observations import find
from clio.providers import sdo as provider


def capture_downloads(monkeypatch):
    download = provider.download_file
    paths = []
    def capture(url, path):
        paths.append(path)
        download(url, path)
    monkeypatch.setattr(provider, 'download_file', capture)
    return paths


def test_download_upload_record_and_remove_tmp(storage, scientific_source, monkeypatch):
    paths = capture_downloads(monkeypatch)
    assert sdo.collect_one(scientific_source.slot, 'aia171', clock=lambda: scientific_source.now) == 'saved'
    row = find('aia_171_nrt')[0]
    assert storage.objects['argus-private', row['key']][0] == scientific_source.content
    assert row['file_id'] == storage.files()[0]['id']
    assert len(storage.objects) == 1
    assert paths and all(not path.exists() and not path.parent.exists() for path in paths)
    assert sdo.collect_one(scientific_source.slot, 'aia171', clock=lambda: scientific_source.now) == 'existing'
    assert scientific_source.download.call_count == 1


def test_interrupted_download_is_discarded(storage, scientific_source, monkeypatch):
    paths = []
    def fail(url, path):
        paths.append(path)
        path.write_bytes(b'incomplete FITS')
        raise OSError('download interrupted')
    monkeypatch.setattr(provider, 'download_file', fail)
    assert sdo.collect_one(scientific_source.slot, 'aia171', clock=lambda: scientific_source.now) == 'failed'
    assert not find('aia_171_nrt') and not storage.objects
    assert all(not path.parent.exists() for path in paths)


@pytest.mark.parametrize('failure', ['upload', 'database'])
def test_failed_storage_cleans_tmp_and_next_attempt_downloads_again(storage, scientific_source, monkeypatch, failure):
    paths = capture_downloads(monkeypatch)
    with monkeypatch.context() as patch:
        if failure == 'upload':
            patch.setattr(storage.client, 'put_object', Mock(side_effect=OSError('S3 unavailable')))
        else:
            patch.setattr(storage.engine, 'begin', Mock(side_effect=SQLAlchemyError('database unavailable')))
        assert sdo.collect_one(scientific_source.slot, 'aia171', clock=lambda: scientific_source.now) == 'failed'
    assert not find('aia_171_nrt') and not storage.objects
    assert all(not path.parent.exists() for path in paths)
    assert sdo.collect_one(scientific_source.slot, 'aia171', clock=lambda: scientific_source.now) == 'saved'
    assert scientific_source.download.call_count == 2
    assert all(not path.parent.exists() for path in paths)
