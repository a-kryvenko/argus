"""Read observation metadata and object references exclusively from PostgreSQL."""
import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from clio.db.models.file_observation import FileObservation
from clio.observations.errors import ObservationUnavailable
from clio.observations.archive import snapshot_receipt
from clio.observations import file_observations, sdo
from common.schemas.observation_files import RawObservationFile
from common.schemas.sdo_images import MissingSDOImage, SDOImageCatalog, SDOImageReference

FILE_KINDS = ('gong', 'goes')


async def load_observation_files(session, as_of, *, start=None, end=None, kinds=('gong', 'goes', 'aia'), channel='aia171'):
    start = start or datetime.min.replace(tzinfo=UTC)
    end = end or datetime.max.replace(tzinfo=UTC)
    items = []
    try:
        for kind in FILE_KINDS:
            if kind not in kinds:
                continue
            rows = await snapshots(session, kind, start=start, end=end, as_of=as_of)
            if len(rows) > 2400:
                raise ObservationUnavailable('Observation file range exceeds the supported batch size')
            items.extend(RawObservationFile(kind=kind, slot_at=row.slot_at, observed_at=row.observed_at,
                available_at=row.available_at, sha256=row.sha256, source_product=row.source_product,
                file=row.file) for row in rows)
        if 'aia' in kinds:
            items.extend(await asyncio.to_thread(aia_files, as_of, start=start, end=end, channel=channel))
    except (SQLAlchemyError, OSError, ValueError, KeyError, TypeError) as exc:
        raise ObservationUnavailable('Observation metadata is unavailable; migrate local files before switching readers') from exc
    return items


def aia_files(as_of, *, start=None, end=None, channel='aia171'):
    lower = max(start or sdo.hourly_slots(as_of)[-1], sdo.hourly_slots(as_of)[-1]).replace(minute=0, second=0, microsecond=0)
    upper = min(end or as_of + timedelta(hours=1), as_of + timedelta(hours=1))
    result = []
    for row in file_observations.find(kind='sdo', variable=channel, entity='fits.original', start=lower, end=upper, as_of=as_of):
        metadata = row['details']
        sdo.validate_metadata(metadata)
        observed = sdo.utc(metadata['observed_at'])
        available = sdo.utc(metadata['available_at'])
        if available > as_of or observed > as_of or (start is not None and observed < start) or (end is not None and observed >= end):
            continue
        result.append(RawObservationFile(kind='aia', slot_at=row['timestamp'], observed_at=observed,
            available_at=available, sha256=metadata['sha256'], source_product=row['source_product'], file=row['file']))
    return result


async def snapshots(session, kind, *, start, end, as_of=None):
    lower = start.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    upper = end.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    if upper < datetime.max.replace(tzinfo=UTC) - timedelta(hours=1):
        upper += timedelta(hours=1)
    statement = select(FileObservation).where(FileObservation.kind == kind,
        FileObservation.error.is_(None), FileObservation.source_key >= lower.isoformat(), FileObservation.source_key < upper.isoformat())
    if as_of is not None:
        statement = statement.where(FileObservation.received_at <= as_of)
    rows = (await session.scalars(statement.order_by(FileObservation.source_key).limit(2401))).all()
    references = await asyncio.to_thread(file_observations.references, [row.file_id for row in rows if row.file_id])
    records = {}
    for row in rows:
        record = snapshot_receipt(row)
        if start <= record['observed_at'] < end and (as_of is None or record['observed_at'] <= as_of):
            reference = references.get(row.file_id)
            if reference is None:
                raise OSError('Observation has no uploaded file reference')
            records[record['slot_at'], record['source_product']] = SimpleNamespace(**record, file=reference)
    return sorted(records.values(), key=lambda row: (row.observed_at, row.slot_at, row.source_product))


def read_image_catalog(start, end, as_of, read_at, channels):
    try:
        images = {(row['timestamp'], row['variable']): row for row in file_observations.find(
            kind='sdo', entity='npz.512', start=start, end=end, as_of=as_of)}
        originals = {(row['timestamp'], row['variable']): row['file'] for row in file_observations.find(
            kind='sdo', entity='fits.original', start=start, end=end, as_of=as_of)}
        items, missing = [], []
        for offset in range(int((end - start).total_seconds() // 3600)):
            slot = start + timedelta(hours=offset)
            for channel in channels:
                row = images.get((slot, channel))
                if row is None:
                    missing.append(MissingSDOImage(slot_at=slot, channel=channel))
                    continue
                metadata = row['details']
                sdo.validate_metadata(metadata)
                if sdo.utc(metadata['slot_at']) != slot or metadata['channel'] != channel:
                    raise ValueError('Observation metadata differs from its registered context')
                if sdo.utc(metadata['available_at']) > as_of or sdo.utc(metadata['observed_at']) > as_of:
                    missing.append(MissingSDOImage(slot_at=slot, channel=channel))
                    continue
                items.append(SDOImageReference(file=row['file'], original=originals.get((slot, channel)), metadata=metadata))
        return SDOImageCatalog(start=start, end=end, as_of=as_of, read_at=read_at, items=items, missing=missing)
    except (SQLAlchemyError, ValueError, KeyError, TypeError) as exc:
        raise ObservationUnavailable('SDO observation metadata is unavailable or corrupt') from exc
