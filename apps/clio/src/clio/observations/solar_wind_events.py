"""Normalize versioned external DONKI/ENLIL data; no Argus forecasting."""
import asyncio
import math
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, text

from clio.db.models.measurement import Measurement
from clio.observations.errors import ObservationUnavailable
from clio.db.models.file_observation import FileObservation
from clio.observations.archive import read_original, read_receipt

from common.schemas.solar_wind_events import (SolarWindEvent, EnlilArrivalInput, ArrivalSourceReceipt,
                                               EventInputs, HourlyWindSpeed, L1Sample, EnlilProfile, SourcePoll)


def timestamp(value):
    if not value or not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return result.astimezone(UTC) if result.tzinfo else None
    except ValueError:
        return None


def positive(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError):
        return None


def published(row, cutoff):
    value = row.get('submissionTime')
    return value is None or (timestamp(value) is not None and timestamp(value) <= cutoff)


def simulation_key(row):
    # Nested CME responses and standalone simulations use the same DONKI link.
    return row.get('link') or row.get('simulationID')


def normalize_event_inputs(inputs):
    cutoff = inputs.as_of
    polls = {}
    for poll in sorted(inputs.polls, key=lambda p: (p.received_at, str(p.poll_id))):
        if poll.received_at <= cutoff:
            polls[poll.endpoint] = poll
    events, simulations = {}, {}
    receipts = {}
    for endpoint, poll in sorted(polls.items()):
        for row in poll.records:
            if not published(row, cutoff):
                continue
            if endpoint == 'CME':
                event_id, onset = row.get('activityID'), timestamp(row.get('startTime'))
                if not event_id or onset is None or onset > cutoff:
                    continue
                events[event_id] = SolarWindEvent(event_id=event_id, kind='cme', status='detected',
                    source='DONKI', source_ids=[event_id], source_url=row.get('link'),
                    available_at=poll.received_at, observed_at=onset)
                analyses = [a for a in row.get('cmeAnalyses') or [] if published(a, cutoff) and positive(a.get('speed')) is not None]
                if analyses:
                    best = max(analyses, key=lambda a: (bool(a.get('isMostAccurate')), timestamp(a.get('submissionTime')) or datetime(1970, 1, 1, tzinfo=UTC)))
                    events[event_id].initial_speed_km_s = positive(best['speed'])
                for analysis in row.get('cmeAnalyses') or []:
                    if not published(analysis, cutoff):
                        continue
                    for simulation in analysis.get('enlilList') or []:
                        key = simulation_key(simulation)
                        if key:
                            receipts.setdefault(key, {})[poll.poll_id] = ArrivalSourceReceipt(poll_id=poll.poll_id, sha256=poll.sha256, received_at=poll.received_at)
                            old = simulations.get(key)
                            ids = set((old[0].get('cmeIDs') or []) if old else [])
                            ids.update(simulation.get('cmeIDs') or [event_id])
                            simulations[key] = ({**simulation, 'cmeIDs': sorted(ids)}, poll.received_at)
            elif endpoint == 'WSAEnlilSimulations':
                for_simulation = simulation_key(row)
                if for_simulation:
                    receipts.setdefault(for_simulation, {})[poll.poll_id] = ArrivalSourceReceipt(poll_id=poll.poll_id, sha256=poll.sha256, received_at=poll.received_at)
                    old = simulations.get(for_simulation)
                    ids = [c['CMEID'] for c in row.get('cmeInputs') or [] if c.get('CMEID')]
                    ids = ids or (old[0].get('cmeIDs', []) if old else [])
                    received = max(poll.received_at, old[1]) if old and not row.get('cmeInputs') else poll.received_at
                    simulations[for_simulation] = ({**row, 'cmeIDs': ids}, received)
            elif endpoint in ('HSS', 'IPS'):
                event_id = row.get('hssID') if endpoint == 'HSS' else row.get('activityID')
                onset = timestamp(row.get('eventTime'))
                # HSS currently has no location field; require an identified L1 instrument.
                near_earth = row.get('location') in ('Earth', 'L1')
                if endpoint == 'HSS' and row.get('location') is None:
                    near_earth = any(i.get('displayName', '').split(':')[0] in
                        ('ACE', 'DSCOVR', 'Wind', 'SOLAR-1', 'IMAP') for i in row.get('instruments') or [])
                if not event_id or onset is None or onset > cutoff or not near_earth:
                    continue
                linked = [x['activityID'] for x in row.get('linkedEvents') or [] if x.get('activityID')]
                events[event_id] = SolarWindEvent(event_id=event_id,
                    kind='hss' if endpoint == 'HSS' else 'interplanetary_shock', status='observed',
                    source='DONKI', source_ids=linked, source_url=row.get('link'),
                    available_at=poll.received_at, observed_at=onset)

    # Retain external alternatives and retractions without selecting a version.
    predictions = []
    for key, (row, received) in sorted(simulations.items()):
        completed = timestamp(row.get('modelCompletionTime'))
        if completed is None or completed > cutoff:
            continue
        arrival = timestamp(row.get('estimatedShockArrivalTime'))
        invalid = ('invalid_arrival_time' if row.get('estimatedShockArrivalTime') is not None and arrival is None
                   else 'arrival_precedes_completion' if arrival is not None and arrival < completed else None)
        predictions.append(EnlilArrivalInput(simulation_id=key,
            cme_ids=sorted(set(row.get('cmeIDs') or [])),
            available_at=max(received, completed), model_completed_at=completed,
            earth_arrival_at=arrival, source_url=row.get('link'),
            duration_hours=positive(row.get('estimatedDuration')),
            glancing_blow=row.get('isEarthGB'), minor_impact=row.get('isEarthMinorImpact'),
            invalid_reason=invalid,
            receipts=sorted(receipts[key].values(), key=lambda r: (r.received_at, str(r.poll_id)))))
    observations = sorted(events.values(), key=lambda e: (e.observed_at or cutoff, e.event_id))
    return observations, predictions


def normalize_events(inputs):
    """Normalize catalog observations only."""
    return normalize_event_inputs(inputs)[0]


async def hourly_speeds(session, as_of):
    """Closed causal hours; prefer >=30 native minutes, otherwise observed hourly sources."""
    rows = (await session.execute(text("""
        WITH eligible AS (
            SELECT observed_at, received_at, value, source_product,
                   date_trunc('hour', observed_at) AS bucket,
                   source_product = 'swpc.rtsw_plasma' AS native
            FROM measurement
            WHERE metric='v' AND observed_at >= :start
              AND observed_at < date_trunc('hour', CAST(:cutoff AS timestamptz))
              AND received_at <= :cutoff
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
    try:
        polls, attempts = await donki_inputs(session, as_of)
    except (OSError, ValueError, KeyError):
        raise ObservationUnavailable('DONKI original or receipt is unavailable or corrupt') from None
    rows = (await session.execute(select(Measurement)
        .where(Measurement.metric.in_(['bx', 'by', 'bz', 'v', 'n', 't']),
               Measurement.source_product.in_(['swpc.rtsw_mag', 'swpc.rtsw_plasma']),
               Measurement.observed_at >= as_of - timedelta(hours=6),
               Measurement.observed_at <= as_of, Measurement.received_at <= as_of)
        .order_by(Measurement.observed_at, Measurement.metric).limit(25001))).scalars().all()
    if len(rows) > 25000:
        raise ObservationUnavailable('L1 input range exceeds supported size')
    speed_hourly = await hourly_speeds(session, as_of)
    result = EventInputs(schema_version=2, as_of=as_of, read_at=now,
        speed_hourly=speed_hourly,
        polls=polls, latest_attempts=attempts,
        l1_samples=[L1Sample(observed_at=p.observed_at,
            available_at=max(p.observed_at, p.received_at),
            metric=p.metric, value=p.value, quality=p.quality)
            for p in rows])

    result.events, result.enlil_arrivals = normalize_event_inputs(result)
    try:
        profile, attempt = await enlil_input(session, as_of)
    except (OSError, ValueError, KeyError):
        raise ObservationUnavailable('ENLIL original or receipt is unavailable or corrupt') from None
    if profile:
        result.enlil_profiles = [profile]
    future = profile and any(p.valid_time > as_of for p in result.enlil_profiles[0].points)
    result.enlil_profile_status = dict(state='available' if future else 'stale' if profile else 'unavailable',
        model_created_at=profile.model_created_at.isoformat() if profile else None,
        latest_attempt=attempt,
        provider='NOAA/SWPC ENLIL', linked_to_donki=False)
    return result


async def read_event_inputs(session, as_of, now):
    try:
        await session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'))
        await session.execute(text("SET LOCAL statement_timeout = '30s'"))
        return await load_event_inputs(session, as_of, now)
    finally:
        await session.rollback()


def _latest(rows, key):
    result = {}
    for row in rows:
        previous = result.get(key(row))
        if previous is None or (row.received_at, row.id) >= (previous.received_at, previous.id):
            result[key(row)] = row
    return result


async def donki_inputs(session, as_of):
    new = (await session.scalars(select(FileObservation).where(FileObservation.kind == 'donki',
        FileObservation.received_at <= as_of, FileObservation.error.is_(None))
        .distinct(FileObservation.source_product)
        .order_by(FileObservation.source_product, FileObservation.received_at.desc(), FileObservation.id.desc()))).all()
    candidates = _latest(new, lambda r: r.source_product)

    def decode():
        from clio.providers.donki import decode as decode_raw
        polls = []
        for endpoint, row in sorted(candidates.items()):
            metadata = read_receipt(row)
            records = decode_raw(read_original('donki', row.raw_path, row.sha256))
            start, end = metadata['request_start'], metadata['request_end']
            polls.append(SourcePoll(endpoint=endpoint, poll_id=row.id, received_at=row.received_at,
                sha256=row.sha256, request_start=start, request_end=end, records=records))
        return polls

    polls = await asyncio.to_thread(decode)
    attempts = await latest_attempts(session, 'donki', as_of)
    return polls, {endpoint: dict(received_at=row.received_at.isoformat(), error=row.error)
                   for endpoint, row in attempts.items()}


async def latest_attempts(session, kind, as_of):
    new = (await session.execute(select(FileObservation.source_product.label('source'), FileObservation.id,
        FileObservation.received_at, FileObservation.error)
        .where(FileObservation.kind == kind, FileObservation.received_at <= as_of)
        .distinct(FileObservation.source_product)
        .order_by(FileObservation.source_product, FileObservation.received_at.desc(), FileObservation.id.desc()))).all()
    return _latest(new, lambda r: r.source if kind == 'donki' else kind)


async def enlil_input(session, as_of):
    new = (await session.scalars(select(FileObservation).where(FileObservation.kind == 'enlil',
        FileObservation.received_at <= as_of, FileObservation.error.is_(None)))).all()

    def choose():
        candidates = []
        for row in new:
            metadata = read_receipt(row)
            created = datetime.fromisoformat(metadata['model_created_at'])
            modified = datetime.fromisoformat(metadata['source_modified_at'])
            if created <= as_of and modified <= as_of:
                candidates.append((created, row.received_at, row.id, row, metadata))
        if not candidates:
            return None
        _, _, _, row, metadata = max(candidates, key=lambda c: c[:3])
        read_original('enlil', row.raw_path, row.sha256)
        return EnlilProfile(receipt_id=row.id, received_at=row.received_at, sha256=row.sha256,
            **{key: metadata[key] for key in ('source_key', 'source_url', 'source_modified_at', 'model_created_at')},
            **metadata['payload'])

    profile = await asyncio.to_thread(choose)
    attempt = (await latest_attempts(session, 'enlil', as_of)).get('enlil')
    return profile, dict(received_at=attempt.received_at.isoformat(), error=attempt.error) if attempt else None
