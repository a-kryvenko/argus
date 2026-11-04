from common.data import datamanager
from datetime import UTC, datetime, timedelta
import hashlib
import asyncio
import io
from pathlib import Path
from unittest.mock import Mock

from astropy.io import fits
import numpy as np
import pytest
from sqlalchemy.exc import SQLAlchemyError

from clio.ingestion.extras import sdo as collector
from clio.ingestion.utils import file_observations
from clio.ingestion.extras import sdo
from clio.routers.observation_files import load_observation_files

SLOT = datetime(2026, 11, 2, 10, tzinfo=UTC)
NOW = SLOT + timedelta(minutes=30)


def test_scientific_collection_writes_objects_and_db_metadata_without_local_archive(storage, monkeypatch, scientific_source):
    native, download = scientific_source.content, scientific_source.download
    monkeypatch.setattr(Path, 'write_bytes', Mock(side_effect=AssertionError('No local observation files')))
    assert collector.collect_one(SLOT, 'aia171', clock=lambda: NOW) == 'saved'
    records = storage.files()
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


def test_catalog_reads_only_database_even_when_storage_is_unavailable(storage, monkeypatch, scientific_source, metadata_session):
    collector.collect_one(SLOT, 'aia171', clock=lambda: NOW)
    read = Mock(side_effect=AssertionError('Catalog must not download objects'))
    monkeypatch.setattr(datamanager, 'read_reference', read)
    monkeypatch.setattr(storage.client, 'get_object', read)
    catalog = asyncio.run(load_observation_files(metadata_session, start=SLOT, end=SLOT+timedelta(hours=1), observation='aia_171_nrt'))
    assert len(catalog) == 1
    item = catalog[0]
    assert item.file.bucket == 'argus-private'
    assert item.file.key.endswith('/original.fits')
    assert item.sha256 == hashlib.sha256(scientific_source.content).hexdigest()
    assert item.observed_at == SLOT
    assert asyncio.run(load_observation_files(metadata_session, start=SLOT, end=SLOT+timedelta(hours=1), observation='hmi_m_nrt')) == []
    read.assert_not_called()


def test_actual_time_is_stored_without_losing_slot_identity(storage, scientific_source):
    image, metadata, original = scientific_source.download.return_value
    actual = SLOT-timedelta(minutes=5)
    metadata['observed_at'] = actual.isoformat()
    assert collector.collect_one(SLOT, 'aia171', clock=lambda: NOW) == 'saved'
    assert collector.collect_one(SLOT, 'aia171', clock=lambda: NOW) == 'existing'
    assert scientific_source.download.call_count == 1
    rows = file_observations.find(kind='sdo', variable='aia171', entity='fits.original')
    assert rows[0]['timestamp'] == actual
    assert rows[0]['details']['slot_at'] == SLOT.isoformat()


def test_db_failure_can_retry_without_replacing_uploaded_objects(storage, monkeypatch, scientific_source):
    register = file_observations.register
    monkeypatch.setattr(file_observations, 'register', Mock(side_effect=SQLAlchemyError('offline')))
    assert collector.collect_one(SLOT, 'aia171', clock=lambda: NOW) == 'failed'
    before = dict(storage.objects)
    monkeypatch.setattr(file_observations, 'register', register)
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
    assert not storage.files()


def test_partial_publication_resumes_legacy_multiline_header_without_replacing_npz(storage, scientific_source, monkeypatch):
    metadata = scientific_source.download.return_value[1]
    metadata['header'].update(COMMENT='First comment\nSecond comment', HISTORY='First step\nSecond step')
    encode = sdo.fits_bytes
    monkeypatch.setattr(sdo, 'fits_bytes', Mock(side_effect=ValueError('Invalid multiline FITS header')))
    assert collector.collect_one(SLOT, 'aia171', clock=lambda: NOW) == 'failed'
    assert {row['entity'] for row in file_observations.find()} == {'npz.512'}
    first_npz = dict(storage.objects)
    monkeypatch.setattr(sdo, 'fits_bytes', encode)
    later = NOW+timedelta(minutes=15)
    storage.clock[0] = later
    assert collector.collect_one(SLOT, 'aia171', clock=lambda: later) == 'saved'
    for key, content in first_npz.items():
        assert storage.objects[key] == content
    prefix = 'images/SDO/aia171/202611021000/'
    with fits.open(io.BytesIO(storage.objects['argus-private', prefix+'512.fits'][0])) as hdus:
        assert list(hdus[0].header['COMMENT']) == ['First comment', 'Second comment']
        assert list(hdus[0].header['HISTORY']) == ['First step', 'Second step']
    assert storage.objects['argus-private', prefix+'original.fits'][0] == scientific_source.content
    assert sdo.scientific_complete(SLOT, 'aia171')
    assert file_observations.find(entity='npz.512')[0]['details']['available_at'] == NOW.isoformat()
