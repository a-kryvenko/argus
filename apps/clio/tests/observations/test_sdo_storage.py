from datetime import UTC, datetime, timedelta
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import Mock

from astropy.io import fits
import numpy as np
import pytest
from sqlalchemy.exc import SQLAlchemyError

from clio.ingestion import sdo_images as collector
from clio.observations import files, file_observations, file_storage, sdo
from clio.observations.file_catalog import read_image_catalog

SLOT = datetime(2026, 11, 2, 10, tzinfo=UTC)
NOW = SLOT + timedelta(minutes=30)


def test_scientific_collection_writes_objects_and_db_metadata_without_local_archive(storage, monkeypatch, scientific_source):
    native, download = scientific_source.content, scientific_source.download
    monkeypatch.setattr(Path, 'write_bytes', Mock(side_effect=AssertionError('No local observation files')))
    assert collector.collect_one(SLOT, 'aia171', clock=lambda: NOW) == 'saved'
    records = files.find()
    assert all(set(row) == {'id', 'timestamp', 'bucket', 'key'} for row in records)
    assert {row['entity'] for row in file_observations.find()} == {'npz.512', 'fits.512', 'fits.original'}
    prefix = 'images/SDO/aia171/202611021000/'
    assert storage.objects['argus-private', prefix+'original.fits'][0] == native
    with fits.open(io.BytesIO(storage.objects['argus-private', prefix+'512.fits'][0])) as hdus:
        assert hdus[0].data.shape == (512, 512)
        assert np.isnan(hdus[0].data[0, 0])
        assert hdus[0].header['CRPIX1'] == 256.5
    assert collector.collect_one(SLOT, 'aia171', clock=lambda: NOW) == 'existing'
    assert download.call_count == 1
    assert not list(storage.root.glob('**/*.npz')) and not list(storage.root.glob('**/*.fits'))


def test_catalog_reads_only_database_even_when_storage_is_unavailable(storage, monkeypatch, scientific_source):
    collector.collect_one(SLOT, 'aia171', clock=lambda: NOW)
    read = Mock(side_effect=AssertionError('Catalog must not download objects'))
    monkeypatch.setattr(file_storage, 'read', read)
    monkeypatch.setattr(storage.client, 'get_object', read)
    catalog = read_image_catalog(SLOT, SLOT+timedelta(hours=1), NOW, NOW, ('aia171', 'hmi_m'))
    assert len(catalog.items) == 1 and len(catalog.missing) == 1
    item = catalog.items[0]
    assert item.file.bucket == 'argus-private' and item.original
    assert item.metadata['channel'] == 'aia171' and 'path' not in item.model_dump()
    read.assert_not_called()


def test_db_failure_can_retry_without_replacing_uploaded_objects(storage, monkeypatch, scientific_source):
    register = files.register
    monkeypatch.setattr(files, 'register', Mock(side_effect=SQLAlchemyError('offline')))
    assert collector.collect_one(SLOT, 'aia171', clock=lambda: NOW) == 'failed'
    before = dict(storage.objects)
    monkeypatch.setattr(files, 'register', register)
    later = NOW + timedelta(minutes=15)
    storage.clock[0] = later
    assert collector.collect_one(SLOT, 'aia171', clock=lambda: later) == 'saved'
    assert file_observations.find(entity='npz.512')[0]['details']['available_at'] == NOW.isoformat()
    for key, value in before.items():
        assert storage.objects[key] == value
    assert sdo.scientific_complete(SLOT, 'aia171')


def test_upload_failure_does_not_register_a_file(storage):
    storage.client.put_object.side_effect = OSError('offline')
    with pytest.raises(OSError):
        sdo.publish(b'npz', 'aia171', SLOT, 'npz.512', details={}, content_type='application/octet-stream')
    assert not files.find()
