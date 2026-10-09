"""Authenticated access to immutable originals, never computed model features."""
import asyncio
import gzip
import hashlib
from datetime import datetime, timedelta
from typing import Literal

from pydantic import AwareDatetime

from fastapi import APIRouter, Depends, HTTPException, Path as PathParameter
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db import get_db_session
from clio.db.models import GONGSnapshot, GOESSnapshot
from clio.observations.files import archive_root
from clio.routers.auth import require_service_token
from common.schemas.forecast_inputs import RawObservationFile

MODELS = {'gong': GONGSnapshot, 'goes': GOESSnapshot}
router = APIRouter(prefix='/internal/v1/observations', dependencies=[Depends(require_service_token)])


async def load_observation_files(session, as_of: datetime):
    files = []
    for kind, model in MODELS.items():
        days = {'gong': 2, 'goes': 90}[kind]
        rows = (await session.scalars(select(model).where(
            model.observed_at >= as_of - timedelta(days=days), model.observed_at <= as_of,
            model.available_at <= as_of).order_by(model.observed_at).limit(2401))).all()
        if len(rows) > 2400:
            raise HTTPException(503, 'Observation file range exceeds the supported batch size')
        files.extend(RawObservationFile(kind=kind, slot_at=row.slot_at, observed_at=row.observed_at,
            available_at=row.available_at, sha256=row.sha256,
            source_product=row.source_product) for row in rows)
    files.extend(await asyncio.to_thread(aia_files, as_of))
    return files


def aia_files(as_of):
    from common.sdo_images import archive_root, hourly_slots, read_metadata, original_path, utc
    root = archive_root()
    files = []
    for slot in hourly_slots(as_of):
        try:
            metadata = read_metadata(root, slot, 'aia193')
        except FileNotFoundError:
            continue
        if (utc(metadata['observed_at']) > as_of or utc(metadata['available_at']) > as_of
                or not original_path(root, slot, 'aia193').is_file()):
            continue
        files.append(RawObservationFile(kind='aia', slot_at=slot,
            observed_at=utc(metadata['observed_at']), available_at=utc(metadata['available_at']),
            sha256=metadata['sha256'], source_product='aia.nrt_193'))
    return files


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
                        session: AsyncSession = Depends(get_db_session),
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
    model = MODELS[kind]
    row = (await session.scalars(select(model).where(model.sha256 == sha256).limit(1))).first()
    if row is None:
        raise HTTPException(404, 'Observation original not found')
    try:
        if row.raw_path:
            root = archive_root(kind).resolve()
            path = (root / row.raw_path).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Original path is outside its archive')
            content = await asyncio.to_thread(path.read_bytes)
        elif kind == 'gong':
            compressed = await session.scalar(select(GONGSnapshot.fits_gzip).where(GONGSnapshot.slot_at == row.slot_at))
            content = await asyncio.to_thread(gzip.decompress, compressed)
        else:
            raise FileNotFoundError()
        if hashlib.sha256(content).hexdigest() != sha256:
            raise ValueError('Original checksum differs from receipt')
    except (OSError, ValueError, TypeError):
        raise HTTPException(503, 'Observation original is unavailable or corrupt') from None
    return Response(content, media_type='application/json' if kind == 'goes' else 'application/fits',
                    headers={'ETag': f'"{sha256}"'})
