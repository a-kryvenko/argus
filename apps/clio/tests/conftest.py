from datetime import UTC, datetime
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from botocore.exceptions import ClientError
import pytest
from sqlalchemy import DateTime, create_engine, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.dialects.sqlite import DATETIME
from sqlalchemy.ext.asyncio import AsyncSession

from common.data import datamanager
from clio.db.models.file import File
from clio.db.models.file_observation import FileObservation
from clio.db import session as db


class SQLiteDateTime(DATETIME):
    """Emulate PostgreSQL timestamptz only in the SQLite test engine."""

    def bind_processor(self, dialect):
        encode = super().bind_processor(dialect)

        def process(value):
            if value is not None and self.timezone:
                if value.tzinfo is None:
                    raise ValueError('Test database timestamps must include a timezone')
                value = value.astimezone(UTC)
            return encode(value)

        return process

    def result_processor(self, dialect, coltype):
        decode = super().result_processor(dialect, coltype)

        def process(value):
            value = decode(value)
            return value.replace(tzinfo=UTC) if value is not None and self.timezone else value

        return process


@pytest.fixture
def object_store(monkeypatch, tmp_path):
    if hasattr(db.get_sync_engine, 'cache_info'):
        db.dispose_sync_engine()
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
    def publish(content, bucket, key, *, content_type):
        uploaded, _ = datamanager.create_immutable_file(content, bucket, key, content_type=content_type)
        with db.get_sync_engine().begin() as connection:
            connection.execute(insert(File).values(timestamp=uploaded, bucket=bucket, key=key)
                .on_conflict_do_nothing(index_elements=['bucket', 'key']))
            reference = dict(connection.execute(select(File.__table__).where(
                File.bucket == bucket, File.key == key)).mappings().one())
        return reference
    yield SimpleNamespace(objects=objects, client=client, clock=clock, root=tmp_path, put=put, publish=publish)
    if hasattr(db.get_sync_engine, 'cache_info'):
        db.dispose_sync_engine()


@pytest.fixture
def storage(monkeypatch, tmp_path, object_store):
    engine = create_engine('sqlite:///' + str(tmp_path / 'registry.sqlite'), connect_args={'check_same_thread': False})
    engine.dialect.colspecs = {**engine.dialect.colspecs, DateTime: SQLiteDateTime}
    File.__table__.create(engine)
    FileObservation.__table__.create(engine)
    monkeypatch.setattr(db, 'get_sync_engine', lambda: engine)
    def registered_files(*, id=None):
        statement = select(File.__table__)
        if id is not None:
            statement = statement.where(File.id == id)
        with engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [dict(row) for row in rows]
    yield SimpleNamespace(**vars(object_store), engine=engine, files=registered_files)
    engine.dispose()


@pytest.fixture
def metadata_session(storage):
    with storage.engine.connect() as connection:
        session = AsyncMock(spec=AsyncSession)
        session.execute.side_effect = connection.execute
        session.scalars.side_effect = connection.scalars
        yield session


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
