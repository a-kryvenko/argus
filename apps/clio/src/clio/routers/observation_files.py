"""Authenticated access to immutable originals, never computed model features."""
import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import AwareDatetime

from fastapi import APIRouter, Depends, HTTPException, Path as PathParameter
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.session import get_request_db_session
from clio.db.models.file_observation import FileObservation
from clio.observations.files import read_original, snapshot_receipt
from clio.routers.auth import require_service_token
from common.schemas.observation_files import RawObservationFile
from common.schemas.response import success_response

FILE_KINDS = ('gong', 'goes')
router = APIRouter(prefix='/internal/v1/observations', dependencies=[Depends(require_service_token)])


async def load_observation_files(session, as_of: datetime, *, start=None, end=None, kinds=('gong', 'goes', 'aia'), channel='aia193'):
    start = start or datetime.min.replace(tzinfo=UTC)
    end = end or datetime.max.replace(tzinfo=UTC)
    files = []
    for kind in FILE_KINDS:
        if kind not in kinds:
            continue
        # Source keys begin with the UTC slot; historical GOES files cover a day.
        lower = start.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        upper = end.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
        if upper < datetime.max.replace(tzinfo=UTC)-timedelta(hours=1):
            upper += timedelta(hours=1)
        rows = (await session.scalars(select(FileObservation).where(
            FileObservation.kind == kind, FileObservation.error.is_(None),
            FileObservation.source_key >= lower.isoformat(), FileObservation.source_key < upper.isoformat(),
            FileObservation.received_at <= as_of).order_by(FileObservation.source_key))).all()
        try:
            receipts = await asyncio.to_thread(lambda: [snapshot_receipt(row) for row in rows])
        except (OSError, ValueError, KeyError):
            raise HTTPException(503, 'Observation file receipt is unavailable or corrupt') from None
        for row in receipts:
            if not start <= row['observed_at'] < end or row['observed_at'] > as_of:
                continue
            files.append(RawObservationFile(kind=kind, slot_at=row['slot_at'], observed_at=row['observed_at'],
                available_at=row['available_at'], sha256=row['sha256'], source_product=row['source_product']))
        if len(files) > 2400:
            raise HTTPException(503, 'Observation file range exceeds the supported batch size')
    if 'aia' in kinds:
        files.extend(await asyncio.to_thread(aia_files, as_of, start=start, end=end, channel=channel))
    return files


def aia_files(as_of, *, start=None, end=None, channel='aia193'):
    from common.sdo_images import archive_root, hourly_slots, read_metadata, original_path, utc
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


@router.get('/files')
async def list_files(
    start: AwareDatetime,
    end: AwareDatetime,
    as_of: AwareDatetime,
    kind: Literal['aia', 'gong', 'goes'],
    channel: Literal['aia171', 'aia193', 'aia211'] = 'aia193',
    session: AsyncSession = Depends(get_request_db_session),
):
    if not timedelta(0) < end - start <= timedelta(days=366) or as_of > datetime.now(UTC):
        raise HTTPException(422, 'Invalid observation file interval or cutoff')
    return success_response(await load_observation_files(session, as_of, start=start, end=end, kinds=(kind,), channel=channel))


def aia_original(sha256, slot_at=None, channel="aia193"):
    from datetime import UTC, datetime
    from common.sdo_images import archive_root, hourly_slots, read_metadata, original_path
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


@router.get('/files/{kind}/{sha256}')
async def original_file(kind: Literal['aia', 'gong', 'goes'],
                        sha256: str = PathParameter(pattern=r'^[0-9a-f]{64}$'),
                        session: AsyncSession = Depends(get_request_db_session),
                        slot_at: AwareDatetime | None = None,
                        channel: Literal["aia171", "aia193", "aia211"] = "aia193"):
    if kind == 'aia':
        try:
            content = await asyncio.to_thread(aia_original, sha256, slot_at, channel)
            if hashlib.sha256(content).hexdigest() != sha256:
                raise ValueError('AIA checksum differs from receipt')
        except FileNotFoundError:
            raise HTTPException(404, 'AIA original is no longer retained') from None
        except (OSError, ValueError):
            raise HTTPException(503, 'AIA original is unavailable or corrupt') from None
        return Response(content, media_type='application/fits', headers={'ETag': f'"{sha256}"'})
    row = (await session.scalars(select(FileObservation).where(
        FileObservation.kind == kind, FileObservation.sha256 == sha256).limit(1))).first()
    if row is None:
        raise HTTPException(404, 'Observation original not found')
    try:
        content = await asyncio.to_thread(read_original, kind, row.raw_path, sha256)
    except (OSError, ValueError, TypeError):
        raise HTTPException(503, 'Observation original is unavailable or corrupt') from None
    return Response(content, media_type='application/json' if kind == 'goes' else 'application/fits',
                    headers={'ETag': f'"{sha256}"'})
