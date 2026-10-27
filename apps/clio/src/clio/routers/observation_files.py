"""HTTP access to immutable observation catalogs and originals."""
from datetime import UTC, datetime, timedelta
from typing import Literal

from common.schemas.response import success_response
from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as PathParameter
from fastapi.responses import Response
from pydantic import AwareDatetime
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.session import get_request_db_session
from clio.observations.errors import ObservationNotFound, ObservationUnavailable
from clio.observations.file_catalog import load_observation_files, load_original

router = APIRouter(prefix='/internal/v1/observations')


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
    try:
        files = await load_observation_files(session, as_of, start=start, end=end, kinds=(kind,), channel=channel)
    except ObservationUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    return success_response(files)


@router.get('/files/{kind}/{sha256}')
async def original_file(kind: Literal['aia', 'gong', 'goes'],
                        sha256: str = PathParameter(pattern=r'^[0-9a-f]{64}$'),
                        session: AsyncSession = Depends(get_request_db_session),
                        slot_at: AwareDatetime | None = None,
                        channel: Literal["aia171", "aia193", "aia211"] = "aia193"):
    try:
        content = await load_original(session, kind, sha256, slot_at=slot_at, channel=channel)
    except ObservationNotFound as exc:
        raise HTTPException(404, str(exc)) from None
    except ObservationUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    media_type = 'application/json' if kind == 'goes' else 'application/fits'
    return Response(content, media_type=media_type, headers={'ETag': f'"{sha256}"'})
