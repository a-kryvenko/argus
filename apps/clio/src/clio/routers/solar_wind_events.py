"""HTTP reads of current event observations."""
from datetime import UTC, datetime, timedelta

from common.schemas.solar_wind_events import EventInputs, HourlyWindSpeed, L1Sample, EnlilProfile, SourcePoll
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from clio.db.session import get_request_db_session
from clio.routers.errors import ObservationUnavailable
from sqlalchemy import func, select, text

from clio.db.models.measurement import Measurement
from clio.db.models.file_observation import FileObservation
from clio.db.models.file import File
from clio.ingestion.extras.solar_wind_events import normalize_event_inputs

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


async def hourly_speeds(session, now):
    """Closed hours; prefer >=30 native minutes, otherwise observed hourly sources."""
    rows = (await session.execute(text("""
        WITH eligible AS (
            SELECT observed_at, received_at, value, source_product,
                   date_trunc('hour', observed_at) AS bucket,
                   source_product = 'swpc.rtsw_plasma' AS native
            FROM measurement
            WHERE metric='v' AND observed_at >= :start
              AND observed_at < date_trunc('hour', CAST(:end AS timestamptz))
              AND value > 0 AND value < 3000
              AND source_product IN ('swpc.rtsw_plasma','omni.hourly','ace.plasma_hourly','soho.plasma_hourly')
        ), native AS (
            SELECT bucket, avg(value) AS value, max(received_at) AS received_at,
                   count(DISTINCT date_trunc('minute', observed_at)) AS sample_count,
                   'swpc.rtsw_plasma' AS source_product
            FROM eligible WHERE native GROUP BY bucket
            HAVING count(DISTINCT date_trunc('minute', observed_at)) >= 30
        ), historical AS (
            SELECT DISTINCT ON (bucket) bucket, value, received_at, 1 AS sample_count, source_product
            FROM eligible WHERE NOT native AND observed_at = bucket
            ORDER BY bucket, received_at DESC, source_product
        )
        SELECT * FROM native UNION ALL
        SELECT * FROM historical WHERE bucket NOT IN (SELECT bucket FROM native)
        ORDER BY bucket
    """), {'start': now-timedelta(days=40), 'end': now})).mappings().all()
    return [HourlyWindSpeed(observed_at=r['bucket'], interval_end=r['bucket']+timedelta(hours=1),
        available_at=max(r['received_at'], r['bucket']+timedelta(hours=1)),
        value=r['value'], source_product=r['source_product'], sample_count=r['sample_count']) for r in rows]


async def load_event_inputs(session, now):
    # Latest *full response* per endpoint handles deletions, corrections and empty feeds.
    try:
        polls, attempts = await donki_inputs(session)
    except (OSError, ValueError, KeyError, TypeError):
        raise ObservationUnavailable('DONKI original or metadata is unavailable or corrupt') from None
    rows = (await session.execute(select(Measurement)
        .where(Measurement.metric.in_(['bx', 'by', 'bz', 'v', 'n', 't']),
               Measurement.source_product.in_(['swpc.rtsw_mag', 'swpc.rtsw_plasma']),
               Measurement.observed_at >= now - timedelta(hours=6),
               Measurement.observed_at <= now)
        .order_by(Measurement.observed_at, Measurement.metric).limit(25001))).scalars().all()
    if len(rows) > 25000:
        raise ObservationUnavailable('L1 input range exceeds supported size')
    speed_hourly = await hourly_speeds(session, now)
    result = EventInputs(schema_version=2, read_at=now,
        speed_hourly=speed_hourly,
        polls=polls, latest_attempts=attempts,
        l1_samples=[L1Sample(observed_at=p.observed_at,
            available_at=max(p.observed_at, p.received_at),
            metric=p.metric, value=p.value)
            for p in rows])

    result.events, result.enlil_arrivals = normalize_event_inputs(result)
    try:
        profile, attempt = await enlil_input(session)
    except (OSError, ValueError, KeyError, TypeError):
        raise ObservationUnavailable('ENLIL original or metadata is unavailable or corrupt') from None
    if profile:
        result.enlil_profiles = [profile]
    future = profile and any(p.valid_time > now for p in result.enlil_profiles[0].points)
    result.enlil_profile_status = dict(state='available' if future else 'stale' if profile else 'unavailable',
        model_created_at=profile.model_created_at.isoformat() if profile else None,
        latest_attempt=attempt,
        provider='NOAA/SWPC ENLIL', linked_to_donki=False)
    return result


async def read_event_inputs(session, now):
    try:
        await session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'))
        await session.execute(text("SET LOCAL statement_timeout = '30s'"))
        return await load_event_inputs(session, now)
    finally:
        await session.rollback()


async def donki_inputs(session):
    latest_ids = select(FileObservation.id, func.row_number().over(
        partition_by=FileObservation.source_product,
        order_by=(FileObservation.observed_at.desc(), FileObservation.id.desc())).label('position')) \
        .where(FileObservation.observation.startswith('donki.', autoescape=True)).subquery()
    rows = (await session.scalars(select(FileObservation)
        .join(latest_ids, latest_ids.c.id == FileObservation.id)
        .where(latest_ids.c.position == 1).order_by(FileObservation.source_product))).all()
    latest = {row.source_product: row for row in rows}
    polls = [SourcePoll(endpoint=endpoint, poll_id=row.id, received_at=row.observed_at,
        sha256=row.sha256, request_start=row.details['request_start'], request_end=row.details['request_end'],
        records=row.details['records']) for endpoint, row in sorted(latest.items())]
    return polls, {endpoint: dict(received_at=row.observed_at.isoformat(), error=None,
        last_success_at=row.observed_at.isoformat()) for endpoint, row in latest.items()}


async def enlil_input(session):
    rows = (await session.execute(select(FileObservation, File.timestamp)
        .join(File, File.id == FileObservation.file_id).where(FileObservation.observation == 'enlil')
        .order_by(FileObservation.observed_at.desc(), File.timestamp.desc()).limit(1))).all()
    if not rows:
        return None, None
    row, uploaded_at = rows[0]
    metadata = row.details
    profile = EnlilProfile(receipt_id=row.id, received_at=uploaded_at, sha256=row.sha256,
        **{key: metadata[key] for key in ('source_key', 'source_url', 'source_modified_at', 'model_created_at')},
        **metadata['payload'])
    return profile, dict(received_at=uploaded_at.isoformat(), error=None)
