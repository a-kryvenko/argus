"""HTTP reads of event observations available at the requested cutoff."""
from datetime import UTC, datetime

from common.schemas.solar_wind_events import EventInputs
from fastapi import APIRouter, Depends, HTTPException
from pydantic import AwareDatetime
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.session import get_request_db_session
from clio.observations.errors import ObservationUnavailable
from clio.observations.solar_wind_events import read_event_inputs

router = APIRouter(prefix='/internal/v1/observations/solar-wind-events', tags=['observations'])


@router.get('/inputs', response_model=EventInputs)
async def inputs(as_of: AwareDatetime, session: AsyncSession = Depends(get_request_db_session)):
    as_of, now = as_of.astimezone(UTC), datetime.now(UTC)
    if as_of > now:
        raise HTTPException(422, 'as_of must not be in the future')
    try:
        return await read_event_inputs(session, as_of, now)
    except ObservationUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
