from datetime import UTC, datetime
import io
from types import SimpleNamespace
from unittest.mock import Mock

from botocore.exceptions import ClientError
import pytest
from sqlalchemy import create_engine

from common.data import datamanager
from clio.db.models.file import File
from clio.db.models.file_observation import FileObservation
from clio.observations import files


@pytest.fixture
def object_store(monkeypatch, tmp_path):
    if hasattr(files.engine, 'cache_info'):
        files.dispose()
    monkeypatch.setenv('S3_PUBLIC_BASE_URL', 'https://media.example.test')
    objects, clock = {}, [datetime(2026, 11, 2, 10, 30, tzinfo=UTC)]
    def put(**kwargs):
        location = kwargs['Bucket'], kwargs['Key']
        assert kwargs['IfNoneMatch'] == '*'
        if location in objects:
            raise ClientError({'Error': {'Code': 'PreconditionFailed'}, 'ResponseMetadata': {'HTTPStatusCode': 412}}, 'PutObject')
        objects[location] = (kwargs['Body'], clock[0], kwargs['ContentType'])
        return {}
    def get(Bucket, Key):
        if (Bucket, Key) not in objects:
            raise ClientError({'Error': {'Code': 'NoSuchKey'}, 'ResponseMetadata': {'HTTPStatusCode': 404}}, 'GetObject')
        return {'Body': io.BytesIO(objects[Bucket, Key][0])}
    client = Mock()
    client.put_object.side_effect = put
    client.head_object.side_effect = lambda Bucket, Key: {'LastModified': objects[Bucket, Key][1]}
    client.get_object.side_effect = get
    client.delete_object.side_effect = lambda Bucket, Key: objects.pop((Bucket, Key), None)
    monkeypatch.setattr(datamanager, 's3_client', client)
    yield SimpleNamespace(objects=objects, client=client, clock=clock, root=tmp_path, put=put)
    if hasattr(files.engine, 'cache_info'):
        files.dispose()


@pytest.fixture
def storage(monkeypatch, tmp_path, object_store):
    engine = create_engine('sqlite:///' + str(tmp_path / 'registry.sqlite'), connect_args={'check_same_thread': False})
    File.__table__.create(engine)
    FileObservation.__table__.create(engine)
    monkeypatch.setattr(files, 'engine', lambda: engine)
    yield SimpleNamespace(**vars(object_store), engine=engine)
    engine.dispose()


@pytest.fixture
def scientific_source(monkeypatch):
    import hashlib
    import numpy as np
    from astropy.io import fits
    from clio.providers import sdo as provider
    slot = datetime(2026, 11, 2, 10, tzinfo=UTC)
    now = datetime(2026, 11, 2, 10, 30, tzinfo=UTC)
    image = np.full((1024, 1024), 4., np.float32)
    image[:2, :2] = np.nan
    header = fits.Header({'CRPIX1': 512.5, 'CDELT1': 2.})
    output = io.BytesIO()
    fits.writeto(output, image, header)
    content = output.getvalue()
    download = Mock(return_value=(image, dict(channel='aia171', header=dict(header), source='https://example.test/aia.fits',
        sha256=hashlib.sha256(content).hexdigest(), observed_at=slot.isoformat(), available_at=now.isoformat()), content))
    monkeypatch.setattr(provider, 'download_observation', download)
    return SimpleNamespace(content=content, download=download, image=image, slot=slot, now=now)
