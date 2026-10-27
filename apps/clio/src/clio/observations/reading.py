"""Prefer file receipts; include legacy observations during the rolling transition."""
import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import select
from sqlalchemy.orm import undefer

from clio.db.models.donki_poll import DonkiPoll
from clio.db.models.enlil_profile import EnlilProfileReceipt
from clio.db.models.file_observation import FileObservation
from clio.db.models.goes_snapshot import GOESSnapshot
from clio.db.models.gong_snapshot import GONGSnapshot
from clio.observations.files import read_original, read_receipt, snapshot_receipt
from common.schemas.solar_wind_events import EnlilProfile, SourcePoll

SNAPSHOTS = {'gong': GONGSnapshot, 'goes': GOESSnapshot}


async def snapshots(session, kind, *, start, end, as_of=None):
    model = SNAPSHOTS[kind]
    legacy = select(model).where(model.observed_at >= start, model.observed_at < end)
    lower = start.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    upper = end.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    if upper < datetime.max.replace(tzinfo=UTC) - timedelta(hours=1):
        upper += timedelta(hours=1)
    archived = select(FileObservation).where(FileObservation.kind == kind,
        FileObservation.error.is_(None), FileObservation.source_key >= lower.isoformat(),
        FileObservation.source_key < upper.isoformat())
    if as_of is not None:
        legacy = legacy.where(model.available_at <= as_of, model.observed_at <= as_of)
        archived = archived.where(FileObservation.received_at <= as_of)
    old = (await session.scalars(legacy.order_by(model.observed_at).limit(2401))).all()
    new = (await session.scalars(archived.order_by(FileObservation.source_key).limit(2401))).all()
    records = {(r.slot_at, r.source_product): SimpleNamespace(**{key: getattr(r, key) for key in (
        'slot_at', 'observed_at', 'available_at', 'sha256', 'source_product', 'raw_path')},
        source_url=getattr(r, 'source_url', None), archived=False) for r in old}
    receipts = await asyncio.to_thread(lambda: [snapshot_receipt(row) for row in new])
    for record in receipts:
        if start <= record['observed_at'] < end and (as_of is None or record['observed_at'] <= as_of):
            records[record['slot_at'], record['source_product']] = SimpleNamespace(**record, archived=True)
    return sorted(records.values(), key=lambda r: (r.observed_at, r.slot_at, r.source_product))


def _latest(rows, key):
    result = {}
    for row in rows:
        previous = result.get(key(row))
        if previous is None or (row.received_at, row.id) >= (previous.received_at, previous.id):
            result[key(row)] = row
    return result


async def donki_inputs(session, as_of):
    old = (await session.scalars(select(DonkiPoll).options(undefer(DonkiPoll.payload))
        .where(DonkiPoll.received_at <= as_of, DonkiPoll.error.is_(None))
        .distinct(DonkiPoll.endpoint)
        .order_by(DonkiPoll.endpoint, DonkiPoll.received_at.desc(), DonkiPoll.id.desc()))).all()
    new = (await session.scalars(select(FileObservation).where(FileObservation.kind == 'donki',
        FileObservation.received_at <= as_of, FileObservation.error.is_(None))
        .distinct(FileObservation.source_product)
        .order_by(FileObservation.source_product, FileObservation.received_at.desc(), FileObservation.id.desc()))).all()
    candidates = _latest([*old, *new], lambda r: r.source_product if isinstance(r, FileObservation) else r.endpoint)

    def decode():
        from clio.providers.donki import decode as decode_raw
        polls = []
        for endpoint, row in sorted(candidates.items()):
            if isinstance(row, FileObservation):
                metadata = read_receipt(row)
                records = decode_raw(read_original('donki', row.raw_path, row.sha256))
                start, end = metadata['request_start'], metadata['request_end']
            else:
                records, start, end = row.payload, row.request_start, row.request_end
            polls.append(SourcePoll(endpoint=endpoint, poll_id=row.id, received_at=row.received_at,
                sha256=row.sha256, request_start=start, request_end=end, records=records))
        return polls

    polls = await asyncio.to_thread(decode)
    attempts = await latest_attempts(session, 'donki', as_of)
    return polls, {endpoint: dict(received_at=row.received_at.isoformat(), error=row.error)
                   for endpoint, row in attempts.items()}


async def latest_attempts(session, kind, as_of):
    model = DonkiPoll if kind == 'donki' else EnlilProfileReceipt
    endpoint = model.endpoint if kind == 'donki' else model.source_url
    old = (await session.execute(select(endpoint.label('source'), model.id, model.received_at, model.error)
        .where(model.received_at <= as_of).distinct(endpoint)
        .order_by(endpoint, model.received_at.desc(), model.id.desc()))).all()
    new = (await session.execute(select(FileObservation.source_product.label('source'), FileObservation.id,
        FileObservation.received_at, FileObservation.error)
        .where(FileObservation.kind == kind, FileObservation.received_at <= as_of)
        .distinct(FileObservation.source_product)
        .order_by(FileObservation.source_product, FileObservation.received_at.desc(), FileObservation.id.desc()))).all()
    return _latest([*old, *new], lambda r: r.source if kind == 'donki' else kind)


async def enlil_input(session, as_of):
    old = (await session.scalars(select(EnlilProfileReceipt).options(undefer(EnlilProfileReceipt.payload))
        .where(EnlilProfileReceipt.received_at <= as_of, EnlilProfileReceipt.error.is_(None),
            EnlilProfileReceipt.model_created_at <= as_of, EnlilProfileReceipt.source_modified_at <= as_of)
        .order_by(EnlilProfileReceipt.model_created_at.desc(), EnlilProfileReceipt.received_at.desc(),
            EnlilProfileReceipt.id.desc()).limit(1))).first()
    new = (await session.scalars(select(FileObservation).where(FileObservation.kind == 'enlil',
        FileObservation.received_at <= as_of, FileObservation.error.is_(None)))).all()

    def choose():
        candidates = []
        if old is not None:
            candidates.append((old.model_created_at, old.received_at, old.id, old, None))
        for row in new:
            metadata = read_receipt(row)
            created = datetime.fromisoformat(metadata['model_created_at'])
            modified = datetime.fromisoformat(metadata['source_modified_at'])
            if created <= as_of and modified <= as_of:
                candidates.append((created, row.received_at, row.id, row, metadata))
        if not candidates:
            return None
        # Prefer the file receipt when the same observation exists in both schemas.
        _, _, _, row, metadata = max(candidates, key=lambda c: (c[0], c[1], c[2], c[4] is not None))
        if metadata is not None:
            read_original('enlil', row.raw_path, row.sha256)
        else:
            metadata = {key: getattr(row, key) for key in (
                'source_key', 'source_url', 'source_modified_at', 'model_created_at', 'payload')}
        return EnlilProfile(receipt_id=row.id, received_at=row.received_at, sha256=row.sha256,
            **{key: metadata[key] for key in ('source_key', 'source_url', 'source_modified_at', 'model_created_at')},
            **metadata['payload'])

    profile = await asyncio.to_thread(choose)
    attempt = (await latest_attempts(session, 'enlil', as_of)).get('enlil')
    return profile, dict(received_at=attempt.received_at.isoformat(), error=attempt.error) if attempt else None
