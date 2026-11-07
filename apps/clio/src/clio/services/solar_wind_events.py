"""Read prepared event observations and bounded numeric histories."""
from datetime import timedelta

from common.schemas.solar_wind_events import EventInputs, HourlyWindSpeed, L1Sample
from sqlalchemy import select, text

from clio.db.models.measurement import Measurement
from clio.db.models.file_observation import FileObservation
from clio.routers.errors import ObservationUnavailable


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
        result = await stored_event_inputs(session, now)
    except (OSError, ValueError, KeyError, TypeError):
        raise ObservationUnavailable('Normalized DONKI observations are unavailable or corrupt') from None
    rows = (await session.execute(select(Measurement)
        .where(Measurement.metric.in_(['bx', 'by', 'bz', 'v', 'n', 't']),
               Measurement.source_product.in_(['swpc.rtsw_mag', 'swpc.rtsw_plasma']),
               Measurement.observed_at >= now - timedelta(hours=6),
               Measurement.observed_at <= now)
        .order_by(Measurement.observed_at, Measurement.metric).limit(25001))).scalars().all()
    if len(rows) > 25000:
        raise ObservationUnavailable('L1 input range exceeds supported size')
    speed_hourly = await hourly_speeds(session, now)
    result.read_at = now
    result.speed_hourly = speed_hourly
    result.l1_samples = [L1Sample(observed_at=p.observed_at,
        available_at=max(p.observed_at, p.received_at), metric=p.metric, value=p.value) for p in rows]

    return result


async def read_event_inputs(session, now):
    try:
        await session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'))
        await session.execute(text("SET LOCAL statement_timeout = '30s'"))
        return await load_event_inputs(session, now)
    finally:
        await session.rollback()


async def stored_event_inputs(session, now):
    observations = ['donki.CME', 'donki.WSAEnlilSimulations', 'donki.HSS', 'donki.IPS']
    sources = (await session.execute(select(FileObservation.__table__).where(
        FileObservation.observation.in_(observations), FileObservation.source_id == '')
        .order_by(FileObservation.source_product))).mappings().all()
    events = (await session.execute(select(FileObservation.details['data']).where(
        FileObservation.observation.in_(observations), FileObservation.details['kind'].as_string() == 'event')
        .order_by(FileObservation.observed_at, FileObservation.source_id))).scalars().all()
    arrivals = (await session.execute(select(FileObservation.details['data']).where(
        FileObservation.observation == 'donki.WSAEnlilSimulations',
        FileObservation.details['kind'].as_string() == 'arrival')
        .order_by(FileObservation.source_id))).scalars().all()
    return EventInputs(schema_version=2, read_at=now, events=events, enlil_arrivals=arrivals,
        polls=[dict(endpoint=s['source_product'], poll_id=s['details']['poll_id'], received_at=s['observed_at'],
            sha256=s['sha256'], request_start=s['details']['request_start'], request_end=s['details']['request_end'],
            record_count=s['details']['record_count']) for s in sources],
        latest_attempts={s['source_product']: dict(received_at=s['observed_at'].isoformat(), error=None,
            last_success_at=s['observed_at'].isoformat()) for s in sources})
