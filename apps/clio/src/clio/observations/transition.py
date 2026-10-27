"""Bridge legacy writers to file receipts during the expand release."""
import asyncio
from datetime import UTC
import gzip
import hashlib
import io
import logging
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from clio.db.models.file_observation import FileObservation
from clio.observations.files import observation_values, store_original, store_receipt

SOURCES = {
    'gong': ('gong_snapshot', ('slot_at',)),
    'goes': ('goes_snapshot', ('slot_at', 'source_product')),
    'donki': ('donki_poll', ('id',)),
    'enlil': ('enlil_profile_receipt', ('id',)),
}


def _identity(kind):
    if kind in ('donki', 'enlil'):
        return 'legacy.id'
    # Stable identities make retries and concurrent old/new writers idempotent.
    return ("md5('" + kind + ":' || to_char(legacy.slot_at AT TIME ZONE 'UTC', "
            "'YYYY-MM-DD\"T\"HH24:MI:SS.US') || ':' || legacy.source_product)::uuid")


def _select(kind, *, missing=False, keyed=False):
    table, keys = SOURCES[kind]
    identity = _identity(kind)
    conditions = []
    if missing:
        conditions.append(f'NOT EXISTS (SELECT 1 FROM clio.file_observation f WHERE f.id = {identity})')
    if keyed:
        conditions.extend(f'legacy.{key} = :{key}' for key in keys)
    where = ' WHERE ' + ' AND '.join(conditions) if conditions else ''
    order = ', '.join('legacy.' + key for key in keys)
    return f'SELECT legacy.*, {identity} AS observation_id FROM clio.{table} legacy{where} ORDER BY {order} LIMIT :limit'


def _original(row, kind):
    from clio.observations.files import archive_path
    if row.get('raw_path'):
        path = archive_path(kind, row['raw_path'])
        if path.is_file():
            return path.read_bytes()
    compressed = row.get('fits_gzip') if kind == 'gong' else row.get('raw_gzip')
    if compressed is None:
        return None
    from clio.observations.files import MAX_ORIGINAL_BYTES
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        raw = stream.read(MAX_ORIGINAL_BYTES + 1)
    if len(raw) > MAX_ORIGINAL_BYTES:
        raise ValueError('Legacy original exceeds the size limit')
    return raw



def export_values(kind, row):
    raw = _original(row, kind)
    if raw is not None and hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError(f'{kind}: original checksum differs from receipt')
    if raw is None and not row.get('error'):
        raise ValueError(f'{kind}: source original is missing')
    if kind in ('gong', 'goes'):
        metadata = {key: value for key, value in row.items()
                    if key not in ('fits_gzip', 'raw_path', 'observation_id')}
        record = store_original(kind, row['slot_at'], row['observed_at'], row['source_product'],
            lambda: raw, expected=metadata, compressed=True,
            relative=Path('migrated') / f'{row["slot_at"].astimezone(UTC):%Y/%m/%d}/'
                f'{row["slot_at"].astimezone(UTC):%H%M%S.%f}-{row["source_product"]}.gz')
        return dict(id=row['observation_id'], error=None, **observation_values(kind, record))
    metadata = {key: value for key, value in row.items()
                if key not in ('id', 'raw_gzip', 'sha256', 'error', 'observation_id')}
    key = str(row['id']) if kind == 'donki' else (row['source_key'] or str(row['id']))
    if kind == 'enlil':
        if row['source_modified_at']:
            key += '@' + row['source_modified_at'].isoformat()
        key += '#' + str(row['id'])
    return store_receipt(kind, row['endpoint'] if kind == 'donki' else 'noaa.enlil', key,
        id=row['id'], raw=raw, metadata=metadata, error=row['error'], received_at=row['received_at'])


async def _export(session, kind, row):
    values = await asyncio.to_thread(export_values, kind, row)
    await session.execute(insert(FileObservation).values(**values).on_conflict_do_nothing(index_elements=['id']))


async def mirror_observation(session, kind, **key):
    """Mirror in the writer's transaction, including the legacy receipt identity."""
    row = (await session.execute(sa.text(_select(kind, missing=True, keyed=True)),
                                 {**key, 'limit': 1})).mappings().first()
    if row is not None:
        await _export(session, kind, row)


async def backfill(*, batch_size=32):
    """Rescan missing rows; late old-version commits are picked up next time."""
    from clio.db.session import get_db_session
    counts = {'failed': 0}
    for kind in SOURCES:
        async with get_db_session() as session:
            rows = (await session.execute(sa.text(_select(kind, missing=True)),
                                          {'limit': batch_size})).mappings().all()
            counts[kind] = 0
            for row in rows:
                try:
                    await _export(session, kind, row)
                except (OSError, ValueError, KeyError) as exc:
                    counts['failed'] += 1
                    logging.getLogger(__name__).error('%s %s: %s', kind, row['observation_id'], exc)
                    continue
                await session.commit()
                counts[kind] += 1
    return counts


async def verify():
    """Read-only release gate: coverage, identity, receipt times and file integrity."""
    from clio.db.session import get_db_session
    from clio.observations.files import read_original, read_receipt
    failures = {}
    async with get_db_session() as session:
        for kind in SOURCES:
            table, _ = SOURCES[kind]
            identity = _identity(kind)
            missing = await session.scalar(sa.text(f'''
                SELECT count(*) FROM clio.{table} legacy
                LEFT JOIN clio.file_observation f ON f.id = {identity}
                WHERE f.id IS NULL OR f.kind <> :kind
                   OR f.received_at <> legacy.{'available_at' if kind in ('gong', 'goes') else 'received_at'}
                   OR f.sha256 IS DISTINCT FROM legacy.sha256
                   OR f.error IS DISTINCT FROM {'NULL' if kind in ('gong', 'goes') else 'legacy.error'}
            '''), {'kind': kind})
            failures[kind] = missing
        # Stream instead of loading the archive into memory.
        rows = await session.stream_scalars(sa.select(FileObservation).execution_options(yield_per=32))
        async for row in rows:
            if row.raw_path is not None:
                try:
                    await asyncio.to_thread(read_original, row.kind, row.raw_path, row.sha256)
                    await asyncio.to_thread(read_receipt, row)
                except (OSError, ValueError, KeyError):
                    failures[row.kind] = failures.get(row.kind, 0) + 1
    return failures
