"""Observation context and metadata for uploaded files."""
from datetime import UTC
import json

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from clio.db.models.file import File
from clio.db.models.file_observation import FileObservation
from clio.observations import files


def register(*, variable, timestamp, entity, bucket, key, uploaded_at, kind, source_product, details=None):
    if timestamp.tzinfo is None:
        raise ValueError('Observation timestamp must include a timezone')
    timestamp = timestamp.astimezone(UTC)
    with files.engine().begin() as connection:
        file = files.register(timestamp=uploaded_at, bucket=bucket, key=key, connection=connection)
        connection.execute(insert(FileObservation).values(file_id=file['id'], variable=variable,
            observed_at=timestamp, entity=entity, kind=kind, source_product=source_product,
            source_key=f'{file["id"]}:{variable}:{timestamp.isoformat()}:{entity}',
            received_at=file['timestamp'], details=json.loads(json.dumps(details, default=str)) if details is not None else None).on_conflict_do_nothing(
                index_elements=['file_id', 'variable', 'observed_at', 'entity']))
    return file


def find(*, kind=None, source_product=None, variable=None, entity=None, bucket=None, key=None,
         start=None, end=None, as_of=None, limit=10000):
    statement = select(FileObservation.id, FileObservation.variable, FileObservation.entity,
        FileObservation.observed_at.label('timestamp'), FileObservation.kind, FileObservation.source_product,
        FileObservation.source_key, FileObservation.details, FileObservation.received_at,
        File.id.label('file_id'), File.timestamp.label('file_timestamp'), File.bucket, File.key
    ).join(File, File.id == FileObservation.file_id).where(FileObservation.error.is_(None))
    for column, value in ((FileObservation.kind, kind), (FileObservation.source_product, source_product),
                          (FileObservation.variable, variable), (FileObservation.entity, entity),
                          (File.bucket, bucket), (File.key, key)):
        if value is not None:
            statement = statement.where(column == value)
    if start is not None:
        statement = statement.where(FileObservation.observed_at >= start)
    if end is not None:
        statement = statement.where(FileObservation.observed_at < end)
    if as_of is not None:
        statement = statement.where(FileObservation.received_at <= as_of)
    with files.engine().connect() as connection:
        rows = connection.execute(statement.order_by(FileObservation.observed_at.desc(), File.key).limit(limit)).mappings().all()
    result = []
    for row in rows:
        timestamp = row['timestamp']
        if timestamp is not None and timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        result.append(dict(id=row['id'], variable=row['variable'], timestamp=timestamp, entity=row['entity'],
            kind=row['kind'], source_product=row['source_product'], source_key=row['source_key'],
            details=row['details'], received_at=row['received_at'],
            file=files.normalize(dict(id=row['file_id'],
                timestamp=row['file_timestamp'], bucket=row['bucket'], key=row['key']))))
    return result


def source(kind, source_key):
    with files.engine().connect() as connection:
        row = connection.execute(select(FileObservation.__table__).where(
            FileObservation.kind == kind, FileObservation.source_key == source_key,
            FileObservation.error.is_(None))).mappings().first()
    return dict(row) if row else None


def references(ids):
    if not ids:
        return {}
    with files.engine().connect() as connection:
        rows = connection.execute(select(File.__table__).where(File.id.in_(ids))).mappings().all()
    return {row['id']: files.normalize(dict(row)) for row in rows}


def set_details(kind, variable, timestamp, details):
    from sqlalchemy import update
    with files.engine().begin() as connection:
        connection.execute(update(FileObservation).where(FileObservation.kind == kind,
            FileObservation.variable == variable, FileObservation.observed_at == timestamp)
            .values(details=json.loads(json.dumps(details, default=str))))
