"""Upload completed temporary files and save their S3 references."""
from datetime import UTC, datetime
import hashlib
import json
from mimetypes import guess_type
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from common.data import datamanager
from clio.db import session as db
from clio.db.models.file import File
from clio.db.models.file_observation import FileObservation

def file_state():
    return FileObservation.details['file_state'].as_string()


def file_rows():
    return select(FileObservation.__table__, File.timestamp.label('file_timestamp'), File.bucket, File.key).join(
        File, File.id == FileObservation.file_id)


def file_record(row):
    return {**row, 'file': dict(id=row['file_id'], timestamp=row['file_timestamp'], bucket=row['bucket'], key=row['key'])}


def save(path: Path, *, observation, observed_at, source_product, details=None,
         bucket='argus-private', variants=()):
    if not isinstance(observed_at, datetime) or observed_at.tzinfo is None:
        raise ValueError('Observation time must include a timezone')
    observed_at = observed_at.astimezone(UTC)
    with path.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    variants = tuple(variants)
    filenames = ",".join(variant.name for variant in variants)
    identity = uuid5(NAMESPACE_URL,
                     f'{bucket}/{observation}/{source_product}/{observed_at.isoformat()}/{checksum}/{path.suffix}/{filenames}')
    prefix = f'observations/{observation}/{identity.hex}'
    key = f'{prefix}/original{path.suffix}'
    objects = [(path, key), *((variant, f'{prefix}/{variant.name}') for variant in variants)]
    if len({object_key for _, object_key in objects}) != len(objects):
        raise ValueError('Observation files must have distinct object keys')
    metadata = json.loads(json.dumps(details or {}, default=str))
    if metadata.keys() & {'file_state', 'extra_files'}:
        raise ValueError('Reserved file manifest key')
    if len(objects) > 1:
        metadata['extra_files'] = [object_key for _, object_key in objects[1:]]

    # Reserve the manifest before upload, so cleanup can recover after a crash.
    engine = db.get_sync_engine()
    lookup = file_rows().where(FileObservation.id == identity)
    with engine.begin() as connection:
        insert = postgres_insert if connection.dialect.name == 'postgresql' else sqlite_insert
        connection.execute(insert(File).values(
            id=uuid4(), timestamp=datetime.now(UTC), bucket=bucket, key=key,
        ).on_conflict_do_nothing(index_elements=[File.bucket, File.key]))
        file_id = connection.scalar(select(File.id).where(File.bucket == bucket, File.key == key))
        connection.execute(insert(FileObservation).values(
            id=identity, observation=observation, observed_at=observed_at, source_product=source_product,
            file_id=file_id, sha256=checksum, details={**metadata, 'file_state': 'uploading'},
        ).on_conflict_do_nothing(index_elements=[FileObservation.id]))
        existing = connection.execute(lookup).mappings().one()
    state = (existing['details'] or {}).get('file_state')
    if state is None:
        return file_record(existing)
    if state != 'uploading' or (existing['details'] or {}).get('extra_files') != metadata.get('extra_files'):
        raise ValueError('Cannot publish an observation with a conflicting file manifest')

    for file_path, object_key in objects:
        datamanager.put_immutable_file(
            file_path, bucket, object_key,
            content_type=guess_type(file_path.name)[0] or 'application/octet-stream',
        )

    with engine.begin() as connection:
        changed = connection.execute(update(FileObservation).where(
            FileObservation.id == identity, file_state() == 'uploading',
        ).values(details=metadata))
        if changed.rowcount:
            connection.execute(update(File).where(File.id == file_id).values(timestamp=datetime.now(UTC)))
        published = connection.execute(lookup.where(file_state().is_(None))).mappings().one()
    return file_record(published)


def find(observation, *, start=None, end=None, source_product=None):
    statement = file_rows().where(FileObservation.observation == observation, file_state().is_(None))
    if start is not None:
        statement = statement.where(FileObservation.observed_at >= start)
    if end is not None:
        statement = statement.where(FileObservation.observed_at < end)
    if source_product is not None:
        statement = statement.where(FileObservation.source_product == source_product)
    with db.get_sync_engine().connect() as connection:
        rows = connection.execute(statement.order_by(FileObservation.observed_at.desc(), File.timestamp.desc())).mappings().all()
    return [file_record(row) for row in rows]
