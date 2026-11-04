"""Observation context and metadata for uploaded files."""
from datetime import UTC, datetime
import json

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from clio.db import session as db
from clio.db.models.file import File
from clio.db.models.file_observation import FileObservation


def _metadata(timestamp, details):
    if timestamp.tzinfo is None:
        raise ValueError('Observation timestamp must include a timezone')
    timestamp = timestamp.astimezone(UTC)
    details = dict(details or {})
    observed = details.get('observed_at', timestamp)
    observed = datetime.fromisoformat(observed) if isinstance(observed, str) else observed
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise ValueError('Observation time must include a timezone')
    observed = observed.astimezone(UTC)
    details.update(slot_at=timestamp.isoformat(), observed_at=observed.isoformat())
    return observed, details


def register(*, variable, timestamp, entity, bucket, key, uploaded_at, kind, source_product, details=None):
    observed, details = _metadata(timestamp, details)
    timestamp = timestamp.astimezone(UTC)
    with db.get_sync_engine().begin() as connection:
        if uploaded_at.tzinfo is None:
            raise ValueError('File timestamp must include a timezone')
        connection.execute(insert(File).values(timestamp=uploaded_at.astimezone(UTC), bucket=bucket, key=key)
            .on_conflict_do_nothing(index_elements=['bucket', 'key']))
        file = dict(connection.execute(select(File.__table__).where(File.bucket == bucket, File.key == key)).mappings().one())
        connection.execute(insert(FileObservation).values(file_id=file['id'], variable=variable,
            observed_at=observed, entity=entity, kind=kind, source_product=source_product,
            source_key=f'{file["id"]}:{variable}:{timestamp.isoformat()}:{entity}',
            received_at=file['timestamp'], details=json.loads(json.dumps(details, default=str))).on_conflict_do_nothing(
                index_elements=['file_id', 'variable', 'observed_at', 'entity']))
    return file


def find(*, kind=None, source_product=None, variable=None, entity=None, bucket=None, key=None,
         start=None, end=None, slot_at=None, limit=10000):
    statement = select(FileObservation.id, FileObservation.variable, FileObservation.entity,
        FileObservation.observed_at.label('timestamp'), FileObservation.kind, FileObservation.source_product,
        FileObservation.source_key, FileObservation.details, FileObservation.received_at,
        File.id.label('file_id'), File.timestamp.label('file_timestamp'), File.bucket, File.key
    ).join(File, File.id == FileObservation.file_id)
    for column, value in ((FileObservation.kind, kind), (FileObservation.source_product, source_product),
                          (FileObservation.variable, variable), (FileObservation.entity, entity),
                          (File.bucket, bucket), (File.key, key)):
        if value is not None:
            statement = statement.where(column == value)
    if start is not None:
        statement = statement.where(FileObservation.observed_at >= start)
    if end is not None:
        statement = statement.where(FileObservation.observed_at < end)
    if slot_at is not None:
        statement = statement.where(FileObservation.details['slot_at'].as_string() == slot_at.astimezone(UTC).isoformat())
    with db.get_sync_engine().connect() as connection:
        rows = connection.execute(statement.order_by(FileObservation.observed_at.desc(), File.key).limit(limit)).mappings().all()
    result = []
    for row in rows:
        result.append(dict(id=row['id'], variable=row['variable'], timestamp=row['timestamp'], entity=row['entity'],
            kind=row['kind'], source_product=row['source_product'], source_key=row['source_key'],
            details=row['details'], received_at=row['received_at'],
            file=dict(id=row['file_id'],
                timestamp=row['file_timestamp'],
                bucket=row['bucket'], key=row['key'])))
    return result


def source(kind, source_key):
    with db.get_sync_engine().connect() as connection:
        row = connection.execute(select(FileObservation.__table__).where(
            FileObservation.kind == kind, FileObservation.source_key == source_key)).mappings().first()
    return dict(row) if row else None


def set_details(kind, variable, timestamp, details):
    from sqlalchemy import update
    observed, details = _metadata(timestamp, details)
    with db.get_sync_engine().begin() as connection:
        connection.execute(update(FileObservation).where(FileObservation.kind == kind,
            FileObservation.variable == variable,
            FileObservation.details['slot_at'].as_string() == timestamp.astimezone(UTC).isoformat())
            .values(observed_at=observed, details=json.loads(json.dumps(details, default=str))))
