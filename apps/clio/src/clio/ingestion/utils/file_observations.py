"""Upload completed temporary files and save their S3 references."""
from datetime import UTC, datetime
import hashlib
import json
import logging
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

from common.data import datamanager
from clio.db import session as db
from clio.db.models.file import File
from clio.db.models.file_observation import FileObservation

logger = logging.getLogger(__name__)


def save(path: Path, *, observation, observed_at, source_product, details=None,
         bucket='argus-private', variants=()):
    if not isinstance(observed_at, datetime) or observed_at.tzinfo is None:
        raise ValueError('Observation time must include a timezone')
    with path.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    prefix = f'observations/{observation}/{uuid4().hex}'
    uploaded = []
    try:
        key = datamanager.upload_file(path, bucket, prefix, 'original'+path.suffix)
        uploaded.append(key)
        for variant in variants:
            uploaded.append(datamanager.upload_file(variant, bucket, prefix, variant.name))
        timestamp = datetime.now(UTC)
        file_id, identity = uuid4(), uuid4()
        metadata = dict(details or {})
        if variants:
            metadata['extra_files'] = uploaded[1:]
        with db.get_sync_engine().begin() as connection:
            connection.execute(File.__table__.insert().values(
                id=file_id, timestamp=timestamp, bucket=bucket, key=key))
            connection.execute(FileObservation.__table__.insert().values(
                id=identity, observation=observation, observed_at=observed_at,
                source_product=source_product, file_id=file_id, sha256=checksum,
                details=json.loads(json.dumps(metadata, default=str))))
        return dict(id=identity, observation=observation, observed_at=observed_at,
            source_product=source_product, file_id=file_id, sha256=checksum, details=metadata,
            file=dict(id=file_id, timestamp=timestamp, bucket=bucket, key=key))
    except Exception:
        for key in uploaded:
            try:
                datamanager.delete_file(bucket, key)
            except Exception:
                logger.exception('Cannot delete unrecorded S3 object %s/%s', bucket, key)
        raise


def find(observation, *, start=None, end=None, source_product=None):
    statement = (select(FileObservation.__table__, File.timestamp.label('file_timestamp'), File.bucket, File.key)
        .join(File, File.id == FileObservation.file_id).where(FileObservation.observation == observation))
    if start is not None:
        statement = statement.where(FileObservation.observed_at >= start)
    if end is not None:
        statement = statement.where(FileObservation.observed_at < end)
    if source_product is not None:
        statement = statement.where(FileObservation.source_product == source_product)
    with db.get_sync_engine().connect() as connection:
        rows = connection.execute(statement.order_by(FileObservation.observed_at.desc(), File.timestamp.desc())).mappings().all()
    return [{**row, 'file': dict(id=row['file_id'], timestamp=row['file_timestamp'], bucket=row['bucket'], key=row['key'])}
            for row in rows]
