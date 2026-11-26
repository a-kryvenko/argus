"""Replace current normalized DONKI observations after a successful full poll."""
import asyncio
import hashlib
import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import delete, exists, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from botocore.exceptions import BotoCoreError, ClientError

from common.data import datamanager
from common.schemas.solar_wind_events import EventInputs, SourcePoll
from clio.db import session as db
from clio.db.locks import ingestion_lock, FILE_LOCKS
from clio.db.models.donki_poll_status import DonkiPollStatus
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


def import_legacy(connection):
    """Normalize the latest legacy originals once, without retaining snapshot history."""
    latest_ids = select(FileObservation.id, func.row_number().over(
        partition_by=FileObservation.source_product,
        order_by=(FileObservation.observed_at.desc(), File.timestamp.desc(), FileObservation.id.desc())
    ).label('position')).join(File, File.id == FileObservation.file_id).where(
        FileObservation.observation.in_(OBSERVATIONS), FileObservation.source_id.is_(None),
        FileObservation.details['file_state'].as_string().is_(None)).subquery()
    previous = connection.execute(select(FileObservation.__table__, File.bucket, File.key)
        .join(File, File.id == FileObservation.file_id)
        .join(latest_ids, latest_ids.c.id == FileObservation.id)
        .where(latest_ids.c.position == 1)).mappings().all()
    known = set(connection.execute(select(FileObservation.source_product).where(
        FileObservation.observation.in_(OBSERVATIONS), FileObservation.source_id == '')).scalars())
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


def delete_unreferenced_files(connection, file_ids):
    for offset in range(0, len(file_ids), 1000):
        connection.execute(delete(File).where(File.id.in_(file_ids[offset:offset+1000]),
            ~exists(select(FileObservation.id).where(FileObservation.file_id == File.id))))


def restore_legacy_polls():
    with ingestion_lock('donki', FILE_LOCKS['donki']) as acquired:
        if not acquired:
            return {'status': 'busy'}
        with db.get_sync_engine().begin() as connection:
            import_legacy(connection)
            prune_legacy(connection)
    return {'status': 'complete'}


def prune_legacy(connection):
    # Originals stay in S3; remove retired snapshots and unreferenced registry rows.
    retired = FileObservation.source_id.is_(None) & (
        FileObservation.observation.in_([*OBSERVATIONS, 'enlil']) |
        FileObservation.source_product.in_([*donki.ENDPOINTS, 'noaa.enlil'])) & (
        FileObservation.details['file_state'].as_string().is_(None))
    file_ids = connection.execute(delete(FileObservation).where(retired)
        .returning(FileObservation.file_id)).scalars().all()
    delete_unreferenced_files(connection, file_ids)


def record_attempt(connection, endpoint, attempted_at, error=None):
    attempt = insert(DonkiPollStatus).values(endpoint=endpoint, attempted_at=attempted_at, error=error)
    connection.execute(attempt.on_conflict_do_update(
        index_elements=[DonkiPollStatus.endpoint],
        set_={'attempted_at': attempt.excluded.attempted_at, 'error': attempt.excluded.error},
        where=attempt.excluded.attempted_at >= DonkiPollStatus.attempted_at,
    ))


def record_poll(endpoint, *, raw, metadata, observed_at=None):
    records = donki.decode(raw)
    poll = SourcePoll(endpoint=endpoint, poll_id=uuid4(), received_at=observed_at or datetime.now(UTC),
        sha256=hashlib.sha256(raw).hexdigest(), records=records, record_count=len(records),
        request_start=metadata['request_start'], request_end=metadata['request_end'])
    normalized = normalize_event_inputs(EventInputs(read_at=poll.received_at, polls=[poll]))
    key = f'observations/donki.{endpoint}/{poll.sha256}/original.json'
    latest = select(FileObservation.__table__).where(
        FileObservation.observation == 'donki.' + endpoint, FileObservation.source_id == '')
    with db.get_sync_engine().connect() as connection:
        last = connection.execute(latest).mappings().one_or_none()
        if last and last['observed_at'] > poll.received_at:
            return uuid_from_poll(last)
    uploaded_at = datamanager.put_immutable_file(raw, BUCKET, key, content_type='application/json')
    with db.get_sync_engine().begin() as connection:
        last = connection.execute(latest.with_for_update()).mappings().one_or_none()
        if last and last['observed_at'] > poll.received_at:
            return uuid_from_poll(last)
        connection.execute(insert(File).values(id=uuid4(), timestamp=uploaded_at, bucket=BUCKET, key=key)
            .on_conflict_do_nothing(index_elements=['bucket', 'key']))
        file_id = connection.scalar(select(File.id).where(File.bucket == BUCKET, File.key == key))
        store_poll(connection, poll, file_id=file_id, normalized=normalized)
        record_attempt(connection, endpoint, poll.received_at)
        if last:
            delete_unreferenced_files(connection, [last['file_id']])
    return poll.poll_id


def uuid_from_poll(row):
    return UUID(row['details']['poll_id'])


async def collect(policy, *, now=None):
    with ingestion_lock('donki', FILE_LOCKS['donki']) as acquired:
        if not acquired:
            return {'status': 'busy', 'failed_endpoints': []}
        now = now or datetime.now(UTC)
        start = (now-timedelta(days=policy.lookback_days)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = now.replace(hour=0, minute=0, second=0, microsecond=0)+timedelta(days=1)
        failed = []
        for endpoint in donki.ENDPOINTS:
            try:
                raw, _ = await asyncio.to_thread(donki.fetch, endpoint, start, now)
                await asyncio.to_thread(record_poll, endpoint, raw=raw, observed_at=datetime.now(UTC),
                    metadata=dict(request_start=start, request_end=end, source_url=donki.BASE_URL+endpoint))
            except (OSError, ValueError, donki.requests.RequestException, BotoCoreError, ClientError, SQLAlchemyError) as exc:
                failed.append(endpoint)
                error = type(exc).__name__
                logger.warning('DONKI %s failed: %s', endpoint, error)
                await asyncio.to_thread(record_failure, endpoint, error)
        return {'status': 'partial' if failed else 'complete', 'failed_endpoints': failed}


def record_failure(endpoint, error):
    with db.get_sync_engine().begin() as connection:
        record_attempt(connection, endpoint, datetime.now(UTC), error)
