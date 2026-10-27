"""Read immutable observations exclusively from file receipts."""
import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import select

from clio.db.models.file_observation import FileObservation
from clio.observations.files import read_original, read_receipt, snapshot_receipt
from common.schemas.solar_wind_events import EnlilProfile, SourcePoll

async def snapshots(session, kind, *, start, end, as_of=None):
    lower = start.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    upper = end.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    if upper < datetime.max.replace(tzinfo=UTC) - timedelta(hours=1):
        upper += timedelta(hours=1)
    archived = select(FileObservation).where(FileObservation.kind == kind,
        FileObservation.error.is_(None), FileObservation.source_key >= lower.isoformat(),
        FileObservation.source_key < upper.isoformat())
    if as_of is not None:
        archived = archived.where(FileObservation.received_at <= as_of)
    rows = (await session.scalars(archived.order_by(FileObservation.source_key).limit(2401))).all()
    records = {}
    receipts = await asyncio.to_thread(lambda: [snapshot_receipt(row) for row in rows])
    for record in receipts:
        if start <= record['observed_at'] < end and (as_of is None or record['observed_at'] <= as_of):
            records[record['slot_at'], record['source_product']] = SimpleNamespace(**record)
    return sorted(records.values(), key=lambda r: (r.observed_at, r.slot_at, r.source_product))


def _latest(rows, key):
    result = {}
    for row in rows:
        previous = result.get(key(row))
        if previous is None or (row.received_at, row.id) >= (previous.received_at, previous.id):
            result[key(row)] = row
    return result


async def donki_inputs(session, as_of):
    new = (await session.scalars(select(FileObservation).where(FileObservation.kind == 'donki',
        FileObservation.received_at <= as_of, FileObservation.error.is_(None))
        .distinct(FileObservation.source_product)
        .order_by(FileObservation.source_product, FileObservation.received_at.desc(), FileObservation.id.desc()))).all()
    candidates = _latest(new, lambda r: r.source_product)

    def decode():
        from clio.providers.donki import decode as decode_raw
        polls = []
        for endpoint, row in sorted(candidates.items()):
            metadata = read_receipt(row)
            records = decode_raw(read_original('donki', row.raw_path, row.sha256))
            start, end = metadata['request_start'], metadata['request_end']
            polls.append(SourcePoll(endpoint=endpoint, poll_id=row.id, received_at=row.received_at,
                sha256=row.sha256, request_start=start, request_end=end, records=records))
        return polls

    polls = await asyncio.to_thread(decode)
    attempts = await latest_attempts(session, 'donki', as_of)
    return polls, {endpoint: dict(received_at=row.received_at.isoformat(), error=row.error)
                   for endpoint, row in attempts.items()}


async def latest_attempts(session, kind, as_of):
    new = (await session.execute(select(FileObservation.source_product.label('source'), FileObservation.id,
        FileObservation.received_at, FileObservation.error)
        .where(FileObservation.kind == kind, FileObservation.received_at <= as_of)
        .distinct(FileObservation.source_product)
        .order_by(FileObservation.source_product, FileObservation.received_at.desc(), FileObservation.id.desc()))).all()
    return _latest(new, lambda r: r.source if kind == 'donki' else kind)


async def enlil_input(session, as_of):
    new = (await session.scalars(select(FileObservation).where(FileObservation.kind == 'enlil',
        FileObservation.received_at <= as_of, FileObservation.error.is_(None)))).all()

    def choose():
        candidates = []
        for row in new:
            metadata = read_receipt(row)
            created = datetime.fromisoformat(metadata['model_created_at'])
            modified = datetime.fromisoformat(metadata['source_modified_at'])
            if created <= as_of and modified <= as_of:
                candidates.append((created, row.received_at, row.id, row, metadata))
        if not candidates:
            return None
        _, _, _, row, metadata = max(candidates, key=lambda c: c[:3])
        read_original('enlil', row.raw_path, row.sha256)
        return EnlilProfile(receipt_id=row.id, received_at=row.received_at, sha256=row.sha256,
            **{key: metadata[key] for key in ('source_key', 'source_url', 'source_modified_at', 'model_created_at')},
            **metadata['payload'])

    profile = await asyncio.to_thread(choose)
    attempt = (await latest_attempts(session, 'enlil', as_of)).get('enlil')
    return profile, dict(received_at=attempt.received_at.isoformat(), error=attempt.error) if attempt else None
