"""HTTP access to normalized solar-wind event observations."""
from datetime import UTC, datetime

from common.schemas.solar_wind_events import EventInputs
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from clio.db.session import get_request_db_session
from clio.routers.errors import ObservationUnavailable
from clio.services.solar_wind_events import read_event_inputs

router = APIRouter(prefix='/internal/v1/observations/solar-wind-events', tags=['observations'])


@router.get('/inputs', response_model=EventInputs)
async def inputs(response: Response, session: AsyncSession = Depends(get_request_db_session)):
    response.headers['Cache-Control'] = 'no-store'
    try:
        return await read_event_inputs(session, datetime.now(UTC))
    except ObservationUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    except SQLAlchemyError:
        raise HTTPException(503, 'Event observation storage is unavailable') from None
