"""HTTP access to immutable observation catalogs."""
from datetime import UTC, datetime, timedelta
from typing import Literal

from common.schemas.response import success_response
from fastapi import APIRouter, Depends, HTTPException
from pydantic import AwareDatetime
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.session import get_request_db_session
from clio.observations.errors import ObservationUnavailable
from clio.observations.file_catalog import load_observation_files
from clio.observations.sdo import OBSERVED_CHANNELS

router = APIRouter(prefix='/internal/v1')


@router.get('/files')
async def list_files(
    start: AwareDatetime,
    end: AwareDatetime,
    as_of: AwareDatetime,
    kind: Literal['aia', 'gong', 'goes'],
    channel: str = 'aia171',
    session: AsyncSession = Depends(get_request_db_session),
):
    if channel not in OBSERVED_CHANNELS:
        raise HTTPException(422, 'Unknown SDO observation channel')
    if not timedelta(0) < end - start <= timedelta(days=366) or as_of > datetime.now(UTC):
        raise HTTPException(422, 'Invalid observation file interval or cutoff')
    try:
        files = await load_observation_files(session, as_of, start=start, end=end, kinds=(kind,), channel=channel)
    except ObservationUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    return success_response(files)
