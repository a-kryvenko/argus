"""Read immutable file catalogs and checksum-verified originals."""
import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
import logging
from types import SimpleNamespace
from zipfile import BadZipFile

from sqlalchemy import select

from clio.db.models.file_observation import FileObservation
from clio.observations.errors import ObservationNotFound, ObservationUnavailable
from clio.observations.archive import read_original, snapshot_receipt
from common.schemas.observation_files import RawObservationFile
from common.schemas.sdo_images import MissingSDOImage, SDOImageCatalog, SDOImageReference

logger = logging.getLogger(__name__)
FILE_KINDS = ('gong', 'goes')


async def load_observation_files(session, as_of: datetime, *, start=None, end=None, kinds=('gong', 'goes', 'aia'), channel='aia193'):
    start = start or datetime.min.replace(tzinfo=UTC)
    end = end or datetime.max.replace(tzinfo=UTC)
    files = []
    for kind in FILE_KINDS:
        if kind not in kinds:
            continue
        try:
            rows = await snapshots(session, kind, start=start, end=end, as_of=as_of)
        except (OSError, ValueError, KeyError):
            raise ObservationUnavailable('Observation file receipt is unavailable or corrupt') from None
        if len(rows) > 2400:
            raise ObservationUnavailable('Observation file range exceeds the supported batch size')
        files.extend(RawObservationFile(kind=kind, slot_at=row.slot_at, observed_at=row.observed_at,
            available_at=row.available_at, sha256=row.sha256,
            source_product=row.source_product) for row in rows)
    if 'aia' in kinds:
        files.extend(await asyncio.to_thread(aia_files, as_of, start=start, end=end, channel=channel))
    return files


def aia_files(as_of, *, start=None, end=None, channel='aia193'):
    from clio.observations.sdo_archive import archive_root, hourly_slots, read_metadata, original_path, utc
    root = archive_root()
    files = []
    for slot in hourly_slots(as_of):
        try:
            metadata = read_metadata(root, slot, channel)
        except FileNotFoundError:
            continue
        if (utc(metadata['observed_at']) > as_of or utc(metadata['available_at']) > as_of
                or not original_path(root, slot, channel).is_file()):
            continue
        observed_at = utc(metadata['observed_at'])
        if (start is not None and observed_at < start) or (end is not None and observed_at >= end):
            continue
        files.append(RawObservationFile(kind='aia', slot_at=slot,
            observed_at=utc(metadata['observed_at']), available_at=utc(metadata['available_at']),
            sha256=metadata['sha256'], source_product='aia.nrt_' + channel[3:]))
    return files


def aia_original(sha256, slot_at=None, channel="aia193"):
    from clio.observations.sdo_archive import archive_root, hourly_slots, read_metadata, original_path
    root = archive_root()
    retained = hourly_slots(datetime.now(UTC))
    slots = retained if slot_at is None else [slot_at] if slot_at in retained else []
    for slot in slots:
        try:
            metadata = read_metadata(root, slot, channel)
        except FileNotFoundError:
            continue
        if metadata['sha256'] == sha256:
            return original_path(root, slot, channel).read_bytes()
    raise FileNotFoundError('AIA original is not retained')


async def load_original(session, kind, sha256, *, slot_at=None, channel="aia193"):
    if kind == 'aia':
        try:
            content = await asyncio.to_thread(aia_original, sha256, slot_at, channel)
            if hashlib.sha256(content).hexdigest() != sha256:
                raise ValueError('AIA checksum differs from receipt')
        except FileNotFoundError:
            raise ObservationNotFound('AIA original is no longer retained') from None
        except (OSError, ValueError):
            raise ObservationUnavailable('AIA original is unavailable or corrupt') from None
        return content
    archived = (await session.scalars(select(FileObservation).where(
        FileObservation.kind == kind, FileObservation.sha256 == sha256).limit(1))).first()
    if archived is None:
        raise ObservationNotFound('Observation original not found')
    try:
        content = await asyncio.to_thread(read_original, kind, archived.raw_path, sha256)
    except (OSError, ValueError, TypeError):
        raise ObservationUnavailable('Observation original is unavailable or corrupt') from None
    return content


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


def read_image_catalog(start, end, as_of, read_at, channels):
    from clio.observations.sdo_archive import archive_root, image_path, read_metadata, utc
    root = archive_root().resolve()
    items, missing = [], []
    for offset in range(int((end - start).total_seconds() // 3600)):
        slot = start + timedelta(hours=offset)
        for name in channels:
            try:
                metadata = read_metadata(root, slot, name)
            except FileNotFoundError:
                missing.append(MissingSDOImage(slot_at=slot, channel=name))
                continue
            except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
                logger.exception('Cannot read SDO observation %s at %s', name, slot)
                raise ObservationUnavailable('SDO observation storage is not readable') from None
            if utc(metadata['available_at']) > as_of or utc(metadata['observed_at']) > as_of:
                missing.append(MissingSDOImage(slot_at=slot, channel=name))
                continue
            items.append(SDOImageReference(path=str(image_path(root, slot, name)), metadata=metadata))
    return SDOImageCatalog(start=start, end=end, as_of=as_of, read_at=read_at,
                           items=items, missing=missing)
