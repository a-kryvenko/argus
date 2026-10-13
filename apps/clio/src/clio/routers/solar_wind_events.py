"""Replay only receipts that existed at the requested cutoff; no forecasting here."""
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import AwareDatetime
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

from clio.db import get_db_session
from clio.db.models import Measurement
from clio.db.models.donki_poll import DonkiPoll
from common.schemas.solar_wind_events import EventInputs, HourlyWindSpeed, L1Sample, SourcePoll

router = APIRouter(prefix='/internal/v1/observations/solar-wind-events', tags=['observations'])


async def hourly_speeds(session, as_of):
    """Closed causal hours; prefer >=30 native minutes, otherwise observed hourly sources."""
    rows = (await session.execute(text("""
        WITH eligible AS (
            SELECT observed_at, received_at, value, source_product,
                   date_trunc('hour', observed_at) AS bucket,
                   source_product = 'swpc.rtsw_plasma' AS native
            FROM clio.measurement
            WHERE metric='v' AND observed_at >= :start
              AND observed_at < date_trunc('hour', CAST(:cutoff AS timestamptz))
              AND received_at <= :cutoff
              AND (interval_end IS NULL OR interval_end <= :cutoff)
              AND quality IS DISTINCT FROM 'flagged' AND quality IS DISTINCT FROM 'missing'
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
    """), {'start': as_of-timedelta(days=40), 'cutoff': as_of})).mappings().all()
    return [HourlyWindSpeed(observed_at=r['bucket'], interval_end=r['bucket']+timedelta(hours=1),
        available_at=max(r['received_at'], r['bucket']+timedelta(hours=1)),
        value=r['value'], source_product=r['source_product'], sample_count=r['sample_count']) for r in rows]


async def load_event_inputs(session, as_of, now):
    # Latest *full response* per endpoint handles deletions, corrections and empty feeds.
    polls = (await session.execute(select(DonkiPoll).options(undefer(DonkiPoll.payload))
        .where(DonkiPoll.received_at <= as_of, DonkiPoll.error.is_(None))
        .distinct(DonkiPoll.endpoint)
        .order_by(DonkiPoll.endpoint, DonkiPoll.received_at.desc(), DonkiPoll.id.desc()))).scalars().all()
    attempts = (await session.execute(select(DonkiPoll.endpoint, DonkiPoll.received_at, DonkiPoll.error)
        .where(DonkiPoll.received_at <= as_of).distinct(DonkiPoll.endpoint)
        .order_by(DonkiPoll.endpoint, DonkiPoll.received_at.desc(), DonkiPoll.id.desc()))).all()
    rows = (await session.execute(select(Measurement)
        .where(Measurement.metric.in_(['bx', 'by', 'bz', 'v', 'n', 't']),
               Measurement.source_product.in_(['swpc.rtsw_mag', 'swpc.rtsw_plasma']),
               Measurement.observed_at >= as_of - timedelta(hours=6),
               Measurement.observed_at <= as_of, Measurement.received_at <= as_of)
        .order_by(Measurement.observed_at, Measurement.metric).limit(25001))).scalars().all()
    if len(rows) > 25000:
        raise HTTPException(503, 'L1 input range exceeds supported size')
    speed_hourly = await hourly_speeds(session, as_of)
    return EventInputs(as_of=as_of, read_at=now,
        speed_hourly=speed_hourly,
        polls=[SourcePoll(endpoint=p.endpoint, poll_id=p.id, received_at=p.received_at,
                          sha256=p.sha256, request_start=p.request_start, request_end=p.request_end,
                          records=p.payload) for p in polls],
        latest_attempts={p.endpoint: dict(received_at=p.received_at.isoformat(), error=p.error) for p in attempts},
        l1_samples=[L1Sample(observed_at=p.observed_at,
            available_at=max(p.observed_at, p.received_at, p.interval_end or p.observed_at),
            metric=p.metric, value=p.value, quality=p.quality, spacecraft=p.spacecraft)
            for p in rows if p.interval_end is None or p.interval_end <= as_of])


@router.get('/inputs', response_model=EventInputs)
async def inputs(as_of: AwareDatetime, session: AsyncSession = Depends(get_db_session)):
    as_of, now = as_of.astimezone(UTC), datetime.now(UTC)
    if as_of > now:
        raise HTTPException(422, 'as_of must not be in the future')
    try:
        await session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'))
        await session.execute(text("SET LOCAL statement_timeout = '30s'"))
        return await load_event_inputs(session, as_of, now)
    finally:
        await session.rollback()
