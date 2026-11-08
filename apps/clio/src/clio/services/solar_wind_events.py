"""Read stored DONKI observations."""
from datetime import datetime

from common.schemas.solar_wind_events import EventInputs, SourcePoll
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.models.file_observation import FileObservation
from clio.routers.errors import ObservationUnavailable

DONKI_OBSERVATIONS = ('donki.CME', 'donki.WSAEnlilSimulations', 'donki.HSS', 'donki.IPS')


async def read_event_inputs(session: AsyncSession, now: datetime) -> EventInputs:
    """Read all inputs from one consistent database snapshot."""
    try:
        await session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'))
        await session.execute(text("SET LOCAL statement_timeout = '30s'"))
        return await _load_event_inputs(session, now)
    finally:
        await session.rollback()


async def _load_event_inputs(session: AsyncSession, now: datetime) -> EventInputs:
    # An empty source_id identifies the latest successful poll for an endpoint.
    poll_query = (
        select(FileObservation.__table__)
        .where(
            FileObservation.observation.in_(DONKI_OBSERVATIONS),
            FileObservation.source_id == '',
        )
        .order_by(FileObservation.source_product)
    )
    event_query = (
        select(FileObservation.details['data'])
        .where(
            FileObservation.observation.in_(DONKI_OBSERVATIONS),
            FileObservation.details['kind'].as_string() == 'event',
        )
        .order_by(FileObservation.observed_at, FileObservation.source_id)
    )
    arrival_query = (
        select(FileObservation.details['data'])
        .where(
            FileObservation.observation == 'donki.WSAEnlilSimulations',
            FileObservation.details['kind'].as_string() == 'arrival',
        )
        .order_by(FileObservation.source_id)
    )

    poll_rows = (await session.execute(poll_query)).mappings().all()
    event_rows = (await session.execute(event_query)).scalars().all()
    arrival_rows = (await session.execute(arrival_query)).scalars().all()

    try:
        polls = []
        for row in poll_rows:
            details = row['details']
            polls.append(SourcePoll(
                endpoint=row['source_product'],
                poll_id=details['poll_id'],
                received_at=row['observed_at'],
                sha256=row['sha256'],
                request_start=details['request_start'],
                request_end=details['request_end'],
                record_count=details['record_count'],
            ))

        latest_attempts = {
            poll.endpoint: {
                'received_at': poll.received_at.isoformat(),
                'error': None,
                'last_success_at': poll.received_at.isoformat(),
            }
            for poll in polls
        }
        return EventInputs(
            schema_version=2,
            read_at=now,
            polls=polls,
            events=event_rows,
            enlil_arrivals=arrival_rows,
            latest_attempts=latest_attempts,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ObservationUnavailable('Normalized DONKI observations are unavailable or corrupt') from exc
