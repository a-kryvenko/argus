"""Replace current normalized DONKI observations after a successful full poll."""
import asyncio
import hashlib
import logging
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import delete, exists, func, select
from sqlalchemy.dialects.postgresql import insert

from common.data import datamanager
from common.schemas.solar_wind_events import EventInputs, SourcePoll
from clio.db import session as db
from clio.db.locks import ingestion_lock, FILE_LOCKS
from clio.db.models.file import File
from clio.db.models.file_observation import FileObservation
from clio.ingestion.extras.solar_wind_events import merge_arrivals, normalize_event_inputs
from clio.providers import donki

logger = logging.getLogger(__name__)
BUCKET = 'argus-private'
OBSERVATIONS = ['donki.' + name for name in donki.ENDPOINTS]
ARRIVALS = 'donki.WSAEnlilSimulations'


def upsert(connection, values):
    statement = insert(FileObservation).values(id=uuid4(), **values)
    connection.execute(statement.on_conflict_do_update(index_elements=['observation', 'source_id'],
        set_={name: getattr(statement.excluded, name) for name in values}))


def store_poll(connection, poll, *, file_id, normalized):
    observation = 'donki.' + poll.endpoint
    events, arrivals = normalized
    common = dict(observation=observation, source_product=poll.endpoint, file_id=file_id, sha256=poll.sha256)
    # Empty source_id is the single successful-poll record, including empty responses.
    upsert(connection, dict(common, source_id='', observed_at=poll.received_at,
        details=dict(kind='poll', poll_id=str(poll.poll_id),
            request_start=poll.request_start.isoformat(), request_end=poll.request_end.isoformat(),
            record_count=len(poll.records))))
    connection.execute(delete(FileObservation).where(FileObservation.observation == observation,
        FileObservation.details['kind'].as_string() == 'event',
        FileObservation.source_id.not_in([event.event_id for event in events])))
    for event in events:
        upsert(connection, dict(common, source_id=event.event_id, observed_at=event.observed_at,
            details=dict(kind='event', data=event.model_dump(mode='json'))))

    if poll.endpoint not in ('CME', 'WSAEnlilSimulations'):
        return
    incoming = {arrival.simulation_id: arrival.model_dump(mode='json') for arrival in arrivals}
    existing = connection.execute(select(FileObservation.source_id, FileObservation.details)
        .where(FileObservation.observation == ARRIVALS,
               FileObservation.details['kind'].as_string() == 'arrival')
        .order_by(FileObservation.source_id).with_for_update()).mappings().all()
    contributions = {row['source_id']: dict(row['details']['sources']) for row in existing}
    headers = {row['source_product']: row for row in connection.execute(select(FileObservation.__table__)
        .where(FileObservation.observation.in_(OBSERVATIONS), FileObservation.source_id == '')).mappings()}
    for simulation_id in sorted(contributions.keys() | incoming.keys()):
        sources = contributions.get(simulation_id, {})
        if poll.endpoint not in sources and simulation_id not in incoming:
            continue
        sources.pop(poll.endpoint, None)
        if simulation_id in incoming:
            sources[poll.endpoint] = incoming[simulation_id]
        if not sources:
            connection.execute(delete(FileObservation).where(FileObservation.observation == ARRIVALS,
                FileObservation.source_id == simulation_id))
        else:
            merged = merge_arrivals(sources)
            original = headers['WSAEnlilSimulations' if 'WSAEnlilSimulations' in sources else 'CME']
            upsert(connection, dict(observation=ARRIVALS, source_product='WSAEnlilSimulations',
                source_id=simulation_id, observed_at=merged.model_completed_at,
                file_id=original['file_id'], sha256=original['sha256'],
                details=dict(kind='arrival', sources=sources, data=merged.model_dump(mode='json'))))


def import_legacy(connection, *, replacing):
    """Normalize the latest legacy originals once, without retaining snapshot history."""
    latest_ids = select(FileObservation.id, func.row_number().over(
        partition_by=FileObservation.source_product,
        order_by=(FileObservation.observed_at.desc(), File.timestamp.desc(), FileObservation.id.desc())
    ).label('position')).join(File, File.id == FileObservation.file_id).where(
        FileObservation.observation.in_(OBSERVATIONS), FileObservation.source_id.is_(None)).subquery()
    previous = connection.execute(select(FileObservation.__table__, File.bucket, File.key)
        .join(File, File.id == FileObservation.file_id)
        .join(latest_ids, latest_ids.c.id == FileObservation.id)
        .where(latest_ids.c.position == 1)).mappings().all()
    known = set(connection.execute(select(FileObservation.source_product).where(
        FileObservation.observation.in_(OBSERVATIONS), FileObservation.source_id == '')).scalars()) | {replacing}
    for row in previous:
        endpoint = row['source_product']
        if endpoint in known:
            continue
        content = datamanager.read_file(row['bucket'], row['key'], max_bytes=donki.MAX_BYTES)
        if hashlib.sha256(content).hexdigest() != row['sha256']:
            raise ValueError('DONKI original checksum differs from its receipt')
        poll = SourcePoll(endpoint=endpoint, poll_id=row['id'], received_at=row['observed_at'],
            sha256=row['sha256'], request_start=row['details']['request_start'],
            request_end=row['details']['request_end'], records=donki.decode(content))
        normalized = normalize_event_inputs(EventInputs(read_at=poll.received_at, polls=[poll]))
        store_poll(connection, poll, file_id=row['file_id'], normalized=normalized)


def prune_legacy(connection, previous_file_id):
    # Originals stay in S3; remove retired snapshots and unreferenced registry rows.
    retired = FileObservation.source_id.is_(None) & (
        FileObservation.observation.in_([*OBSERVATIONS, 'enlil']) |
        FileObservation.source_product.in_([*donki.ENDPOINTS, 'noaa.enlil']))
    file_ids = connection.execute(delete(FileObservation).where(retired)
        .returning(FileObservation.file_id)).scalars().all()
    if previous_file_id is not None:
        file_ids.append(previous_file_id)
    for offset in range(0, len(file_ids), 1000):
        connection.execute(delete(File).where(File.id.in_(file_ids[offset:offset+1000]),
            ~exists(select(FileObservation.id).where(FileObservation.file_id == File.id))))


def record_poll(endpoint, *, raw, metadata, observed_at=None):
    records = donki.decode(raw)
    poll = SourcePoll(endpoint=endpoint, poll_id=uuid4(), received_at=observed_at or datetime.now(UTC),
        sha256=hashlib.sha256(raw).hexdigest(), records=records, record_count=len(records),
        request_start=metadata['request_start'], request_end=metadata['request_end'])
    normalized = normalize_event_inputs(EventInputs(read_at=poll.received_at, polls=[poll]))
    key = f'observations/donki.{endpoint}/{poll.sha256}/original.json'
    with db.get_sync_engine().begin() as connection:
        last = connection.execute(select(FileObservation.__table__).where(
            FileObservation.observation == 'donki.' + endpoint, FileObservation.source_id == '')
            .with_for_update()).mappings().one_or_none()
        if last and last['observed_at'] > poll.received_at:
            return SourcePoll.model_validate(dict(last['details'], endpoint=endpoint,
                received_at=last['observed_at'], sha256=last['sha256'])).poll_id
        import_legacy(connection, replacing=endpoint)
        uploaded_at = datamanager.put_immutable_file(raw, BUCKET, key, content_type='application/json')
        connection.execute(insert(File).values(id=uuid4(), timestamp=uploaded_at, bucket=BUCKET, key=key)
            .on_conflict_do_nothing(index_elements=['bucket', 'key']))
        file_id = connection.scalar(select(File.id).where(File.bucket == BUCKET, File.key == key))
        store_poll(connection, poll, file_id=file_id, normalized=normalized)
        prune_legacy(connection, last['file_id'] if last else None)
    return poll.poll_id


async def collect(policy, *, now=None):
    with ingestion_lock('donki', FILE_LOCKS['donki']) as acquired:
        if not acquired:
            return False
        now = now or datetime.now(UTC)
        start = (now-timedelta(days=policy.lookback_days)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = now.replace(hour=0, minute=0, second=0, microsecond=0)+timedelta(days=1)
        failed = []
        for endpoint in donki.ENDPOINTS:
            try:
                raw, _ = await asyncio.to_thread(donki.fetch, endpoint, start, now)
                await asyncio.to_thread(record_poll, endpoint, raw=raw, observed_at=now,
                    metadata=dict(request_start=start, request_end=end, source_url=donki.BASE_URL+endpoint))
            except (OSError, ValueError, donki.requests.RequestException) as exc:
                failed.append(endpoint)
                logger.warning('DONKI %s failed: %s', endpoint, type(exc).__name__)
        if failed:
            raise RuntimeError('DONKI collection failed: '+', '.join(failed))
        return True
