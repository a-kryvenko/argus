"""Track ingestion without losing failure state when observation writes roll back."""
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import logging

import requests
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError

from argus_clio.db.models import ObservationSourceStatus
from argus_clio.services.collection.specs import SOURCE_SPECS, ATTEMPT_TIMEOUT_SECONDS, overdue_after

logger = logging.getLogger(__name__)


def public_error(exc: Exception) -> tuple[str, str]:
    # Never publish exception text: it may contain credentials, SQL or provider bodies.
    if isinstance(exc, requests.Timeout):
        return 'source_timeout', 'The source request timed out.'
    if isinstance(exc, requests.HTTPError):
        code = exc.response.status_code if exc.response is not None else None
        return 'source_http_error', f'The source returned HTTP {code}.' if code else 'The source returned an HTTP error.'
    if isinstance(exc, requests.ConnectionError):
        return 'source_connection_error', 'Could not connect to the source.'
    if isinstance(exc, (requests.exceptions.JSONDecodeError, ValueError)):
        return 'invalid_response', 'The source response could not be parsed or validated.'
    if isinstance(exc, requests.RequestException):
        return 'source_request_error', 'The source request failed.'
    if isinstance(exc, SQLAlchemyError):
        return 'storage_error', 'Could not save observations.'
    return 'collection_error', 'Observation collection failed.'


class Attempt:
    def __init__(self, source_id: str, started: datetime):
        self.source_id = source_id
        self.started = started
        self.response_at = None
        self.observation_at = None
        self.interval_end = None
        self.quality = None

    def received(self, records: list[dict]) -> None:
        self.response_at = datetime.now(UTC)
        if self.source_id.startswith('solar_wind_'):
            point = max((row for row in records if row['active']), key=lambda row: row['observed_at'], default=None)
            if point is not None:
                self.observation_at = point['observed_at']
                usable = sum(value is not None for value in point['values'].values())
                self.quality = ('unavailable' if usable == 0 or point['raw'].get('overall_quality') not in (None, 0)
                                else 'complete' if usable == len(point['values']) else 'partial')
        else:
            point = max(records, key=lambda row: row['interval_start'], default=None)
            if point is not None:
                self.observation_at = point['interval_start']
                self.interval_end = point['interval_end']
                self.quality = 'complete' if point['value'] is not None and point['quality'] != 'flagged' else 'unavailable'
        if point is None:
            self.quality = 'unavailable'

    def response_values(self) -> dict:
        return {'last_response_at': self.response_at, 'latest_observation_at': self.observation_at,
                'latest_interval_end': self.interval_end, 'data_quality': self.quality}


@asynccontextmanager
async def track_attempt(source_id, session, factory):
    """Caller holds the source advisory lock. Observation commit belongs to this context."""
    attempt = Attempt(source_id, datetime.now(UTC))
    # Separate transaction: visible during fetch, survives observation rollback/crash.
    async with factory() as status_session:
        statement = insert(ObservationSourceStatus).values(
            source_id=source_id, last_attempt_at=attempt.started, consecutive_failures=0)
        await status_session.execute(statement.on_conflict_do_update(
            index_elements=['source_id'], set_={'last_attempt_at': attempt.started}))
        await status_session.commit()
    matches = (ObservationSourceStatus.source_id == source_id,
               ObservationSourceStatus.last_attempt_at == attempt.started)
    try:
        yield attempt
        finished = datetime.now(UTC)
        await session.execute(update(ObservationSourceStatus).where(*matches).values(
            last_completed_at=finished, last_success_at=finished, consecutive_failures=0,
            **attempt.response_values()))
        # Atomically record success with the source measurements.
        await session.commit()
    except Exception as exc:
        try:
            await session.rollback()
        except Exception:
            logger.exception('Could not roll back %s observations', source_id)
        code, message = public_error(exc)
        try:
            async with factory() as status_session:
                values = {'last_completed_at': datetime.now(UTC), 'last_error_at': datetime.now(UTC),
                          'last_error_code': code, 'last_error_message': message,
                          'consecutive_failures': ObservationSourceStatus.consecutive_failures + 1}
                if attempt.response_at is not None:
                    values.update(attempt.response_values())
                await status_session.execute(update(ObservationSourceStatus).where(*matches).values(**values))
                await status_session.commit()
        except Exception:
            # A database outage cannot be persisted in that same unavailable database.
            logger.exception('Could not persist %s collection failure', source_id)
        raise


def describe(source_id: str, record: ObservationSourceStatus | None, now: datetime) -> dict:
    spec = SOURCE_SPECS[source_id]
    fields = ('last_attempt_at', 'last_completed_at', 'last_response_at', 'last_success_at',
              'last_error_at', 'last_error_code', 'last_error_message', 'latest_observation_at',
              'latest_interval_end', 'data_quality')
    result = {**spec, 'source_id': source_id, 'overdue_after_seconds': overdue_after(source_id),
              'attempt_timeout_seconds': ATTEMPT_TIMEOUT_SECONDS,
              **{field: getattr(record, field) if record is not None else None for field in fields},
              'consecutive_failures': record.consecutive_failures if record is not None else 0}
    if record is None:
        return {**result, 'status': 'not_started', 'collector_status': 'unknown', 'data_status': 'unknown',
                'attempt_age_seconds': None, 'data_age_seconds': None}
    attempt_age = max(0, (now-record.last_attempt_at).total_seconds())
    running = record.last_completed_at is None or record.last_completed_at < record.last_attempt_at
    collector_status = ('stalled' if running and attempt_age > ATTEMPT_TIMEOUT_SECONDS else
                        'overdue' if attempt_age > overdue_after(source_id) else
                        'collecting' if running else 'running')
    basis = record.latest_interval_end if spec['freshness_basis'] == 'interval_end' else record.latest_observation_at
    data_age = max(0, (now-basis).total_seconds()) if basis is not None else None
    data_status = ('unknown' if record.data_quality is None else 'unavailable' if record.data_quality == 'unavailable'
                   else 'delayed' if data_age is not None and data_age > spec['stale_after_seconds']
                   else 'partial' if record.data_quality == 'partial' else 'fresh')
    status = ('collector_stalled' if collector_status == 'stalled' else
              'collector_overdue' if collector_status == 'overdue' else
              'collection_error' if record.consecutive_failures else
              'collecting' if collector_status == 'collecting' and record.last_success_at is None else
              'source_delayed' if data_status == 'delayed' else
              'data_unavailable' if data_status in ('unavailable', 'unknown') else
              'data_partial' if data_status == 'partial' else 'ok')
    return {**result, 'status': status, 'collector_status': collector_status, 'data_status': data_status,
            'attempt_age_seconds': int(attempt_age), 'data_age_seconds': int(data_age) if data_age is not None else None}


async def source_status(session, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    records = {row.source_id: row for row in (await session.execute(select(ObservationSourceStatus))).scalars()}
    sources = {source_id: describe(source_id, records.get(source_id), now) for source_id in SOURCE_SPECS}
    return {'generated_at': now, 'status': 'ok' if all(source['status'] == 'ok' for source in sources.values()) else 'degraded',
            'sources': sources}
