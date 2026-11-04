"""Publish immutable source observations to S3 and retain their metadata in PostgreSQL."""
from datetime import UTC, datetime
import hashlib
import json
from uuid import UUID, uuid4
from types import SimpleNamespace

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from common.data import datamanager
from clio.db.models.file_observation import FileObservation
from clio.ingestion.utils import file_observations
from clio.db import session as db
from clio.db.models.file import File

MAX_ORIGINAL_BYTES = 128 * 1024 * 1024


def remote_key(kind, timestamp, source, id=None):
    extensions = {'gong': 'fits', 'goes': 'json', 'donki': 'json', 'enlil': 'nc'}
    filename = f'{id}.{extensions[kind]}' if id is not None else f'original.{extensions[kind]}'
    return f'observations/{kind.upper()}/{timestamp.astimezone(UTC):%Y%m%d%H%M}/{source}/{filename}'


def file_fields(kind, record):
    observed = record.get('observed_at')
    observed = datetime.fromisoformat(str(observed)) if observed is not None else None
    return dict(file_id=UUID(str(record['file_id'])), variable=kind if kind in ('gong', 'goes') else None,
        observed_at=observed, entity={'gong': 'fits.original', 'goes': 'json.original',
            'donki': 'json.original', 'enlil': 'netcdf.original'}[kind],
        details=json.loads(json.dumps(record, default=str)))


def publish_original(kind, timestamp, source, content, record, *, id=None):
    content_type = {'gong': 'application/fits', 'goes': 'application/json',
                    'donki': 'application/json', 'enlil': 'application/x-netcdf'}[kind]
    key = (f'observations/DONKI/{source}/{record["sha256"]}.json' if kind == 'donki'
           else remote_key(kind, timestamp, source, id))
    uploaded, _ = datamanager.create_immutable_file(content, 'argus-private', key, content_type=content_type)
    with db.get_sync_engine().begin() as connection:
        connection.execute(insert(File).values(timestamp=uploaded, bucket='argus-private', key=key)
            .on_conflict_do_nothing(index_elements=['bucket', 'key']))
        reference = dict(connection.execute(select(File.__table__).where(
            File.bucket == 'argus-private', File.key == key)).mappings().one())
    record['file_id'] = str(reference['id'])
    return reference


def source_key(slot, source):
    return slot.astimezone(UTC).isoformat() + ':' + source


def store_original(kind, slot, observed, source, download, *, metadata=None):
    existing = file_observations.source(kind, source_key(slot, source))
    if existing is not None:
        return snapshot_receipt(SimpleNamespace(**existing))
    content = download()
    if content is None:
        return None
    if len(content) > MAX_ORIGINAL_BYTES:
        raise ValueError('Observation original exceeds the size limit')
    record = dict(metadata or {}, slot_at=slot.isoformat(), observed_at=observed.isoformat(),
        available_at=datetime.now(UTC).isoformat(), sha256=hashlib.sha256(content).hexdigest(), source_product=source)
    publish_original(kind, slot, source, content, record)
    return {**record, **{name: datetime.fromisoformat(record[name]) for name in ('slot_at', 'observed_at', 'available_at')}}


def observation_values(kind, record):
    identity = f'{kind}:{record["slot_at"].astimezone(UTC):%Y-%m-%dT%H:%M:%S.%f}:{record["source_product"]}'
    return dict(id=UUID(hashlib.md5(identity.encode(), usedforsecurity=False).hexdigest()), kind=kind,
        source_product=record['source_product'], source_key=source_key(record['slot_at'], record['source_product']),
        received_at=record['available_at'], **file_fields(kind, record))


def read_receipt(row):
    if not row.file_id or not row.details:
        raise OSError('Observation file reference or metadata is missing')
    return dict(row.details)


def snapshot_receipt(row):
    record = read_receipt(row)
    return {**record, **{name: datetime.fromisoformat(str(record[name])) for name in ('slot_at', 'observed_at')},
            'available_at': row.received_at, 'source_product': row.source_product, 'file_id': row.file_id}


def store_receipt(kind, source_product, key, *, raw, metadata=None, received_at=None, id=None):
    id = id or uuid4()
    received_at = received_at or datetime.now(UTC)
    values = dict(id=id, kind=kind, source_product=source_product, source_key=key,
                  received_at=received_at)
    if raw is None:
        raise ValueError('A successful file observation requires an original')
    if len(raw) > MAX_ORIGINAL_BYTES:
        raise ValueError('Observation original exceeds the size limit')
    receipt = dict(metadata or {}, sha256=hashlib.sha256(raw).hexdigest(), received_at=received_at.isoformat())
    if kind == 'donki':
        from clio.providers.donki import decode
        receipt['records'] = decode(raw)
    publish_original(kind, received_at, source_product, raw, receipt, id=id)
    return {**values, **file_fields(kind, receipt)}


async def snapshots(session, kind, *, start, end, limit=None):
    statement = select(FileObservation).where(FileObservation.kind == kind,
        FileObservation.observed_at >= start, FileObservation.observed_at < end)
    statement = statement.order_by(FileObservation.observed_at, FileObservation.received_at, FileObservation.id)
    if limit is not None:
        statement = statement.limit(limit + 1)
    rows = (await session.scalars(statement)).all()
    if limit is not None and len(rows) > limit:
        raise ValueError('Observation file range exceeds the supported batch size')
    records = {}
    for row in rows:
        record = snapshot_receipt(row)
        records[record['slot_at'], record['source_product']] = SimpleNamespace(**record)
    return sorted(records.values(), key=lambda row: (row.observed_at, row.slot_at, row.source_product))
