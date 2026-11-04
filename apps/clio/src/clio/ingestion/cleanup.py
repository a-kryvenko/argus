import asyncio
from contextlib import ExitStack
from datetime import datetime, timedelta
import logging

from botocore.exceptions import BotoCoreError, ClientError

from sqlalchemy import select, delete, exists, or_, and_

from common.data import datamanager
from clio.db import session as db
from clio.db.locks import ingestion_lock, FILE_LOCKS
from clio.db.models.file import File
from clio.db.models.file_observation import FileObservation as Receipt
from clio.db.models.measurement import Measurement
from clio.db.models.normalized_observation import NormalizedObservation
from clio.db.session import get_db_session
from clio.ingestion.utils.products import PRODUCTS, OBSERVATIONS

logger = logging.getLogger(__name__)
MAX_FILE_DELETIONS = 1000


async def cleanup(config, now: datetime):
    products = [name for name, product in PRODUCTS.items() if product.kind == 'numeric' and 'live' in product.modes]
    
    async with get_db_session() as session:
        # Published means can describe an older window while still being the
        # latest release. Retain them for their configured collection horizon.
        means = {m: p.backfill_days for m, p in config.observations.items() if m.endswith('_81mean')}
        expired = or_(and_(Measurement.metric.not_in(list(means)),
                           Measurement.observed_at < now - timedelta(days=config.retention.live_days)),
                      *(and_(Measurement.metric == m,
                             Measurement.observed_at < now - timedelta(days=max(days, config.retention.live_days)))
                        for m, days in means.items()))
        live = await session.execute(delete(Measurement).where(
            Measurement.source_product.in_(products),
            expired))
        normalized = await session.execute(delete(NormalizedObservation).where(
            NormalizedObservation.observed_at < now - timedelta(days=config.retention.normalized_days)))
        await session.commit()
    
    result = {'live_removed': live.rowcount, 'normalized_removed': normalized.rowcount}
    result['files'] = await asyncio.to_thread(cleanup_files, config, now)
    if any(report['failed'] for report in result['files'].values()):
        result['status'] = 'partial'
    
    return result


def remove_unreferenced(bucket, prefix, *, now, max_files=MAX_FILE_DELETIONS):
    """Caller must hold every collection lock for this prefix."""
    counts = dict(files_deleted=0, failed=0)
    referenced = exists(select(Receipt.id).where(Receipt.file_id == File.id))
    with db.get_sync_engine().connect() as connection:
        unused = connection.execute(select(File.__table__).where(File.bucket == bucket,
            File.key.startswith(prefix, autoescape=True), ~referenced,
            File.timestamp < now-timedelta(minutes=15))
            .order_by(File.timestamp, File.id).limit(max_files)).mappings().all()
    for reference in unused:
        try:
            datamanager.delete_file(reference['bucket'], reference['key'])
        except (BotoCoreError, ClientError, OSError):
            counts['failed'] += 1
            logger.exception('Cannot remove unused object %s/%s', reference['bucket'], reference['key'])
            continue
        # Leave the registry entry on S3 failure, so the next cycle can retry.
        with db.get_sync_engine().begin() as connection:
            connection.execute(delete(File).where(File.id == reference['id'], ~referenced))
        counts['files_deleted'] += 1
    return counts


def _expire(kind, condition):
    with db.get_sync_engine().begin() as connection:
        return connection.execute(delete(Receipt).where(Receipt.kind == kind, condition)).rowcount


def _enlil(now, days):
    cutoff = now-timedelta(days=days)
    with db.get_sync_engine().connect() as connection:
        old = connection.execute(select(Receipt.__table__).where(
            Receipt.kind == 'enlil', Receipt.received_at < cutoff)).mappings().all()
    keep, versions = [], []
    if old:
        keep.append(max(old, key=lambda row: (row['received_at'], row['id']))['id'])
    for row in old:
        metadata = row['details'] or {}
        try:
            created, modified = (datetime.fromisoformat(metadata[name])
                                 for name in ('model_created_at', 'source_modified_at'))
            if created.tzinfo is None or modified.tzinfo is None:
                raise ValueError('ENLIL metadata timestamps must include a timezone')
        except (KeyError, TypeError, ValueError):
            keep.append(row['id'])  # Unknown metadata must not cause loss of the only successful profile.
            continue
        if created <= cutoff and modified <= cutoff:
            versions.append((created, row['received_at'], row['id']))
        else:
            keep.append(row['id'])
    if versions:
        keep.append(max(versions)[2])
    return _expire('enlil', and_(Receipt.received_at < cutoff, Receipt.id.not_in(keep)))


def cleanup_files(config, now, *, max_files=MAX_FILE_DELETIONS):
    from clio.ingestion.extras.donki import cleanup as cleanup_donki
    report = {}
    for observation, policy in config.observations.items():
        definition = OBSERVATIONS[observation]
        if definition.kind != 'file':
            continue
        kind = definition.file_kind or observation
        lanes, bucket, prefix = {
            'sdo': (('sdo-live', 'sdo-warmup'), 'argus-private', 'images/SDO/'),
            'sdo_browse': (('sdo-browse',), 'argus-public', 'images/AIA/'),
            'gong': (('gong',), 'argus-private', 'observations/GONG/'),
            'goes': (('goes',), 'argus-private', 'observations/GOES/'),
        }[kind]
        with ExitStack() as stack:
            if not all(stack.enter_context(ingestion_lock(lane, FILE_LOCKS[lane])) for lane in lanes):
                report[observation] = dict(busy=True, expired=0, files_deleted=0, failed=0)
                continue
            cutoff = now - timedelta(days=policy.retention_days)
            condition = or_(Receipt.observed_at < cutoff,
                and_(Receipt.observed_at.is_(None), Receipt.received_at < cutoff))
            if definition.channel:
                condition = and_(condition, Receipt.variable == definition.channel)
            expired = _expire(kind, condition)
            report[observation] = dict(expired=expired,
                **remove_unreferenced(bucket, prefix, now=now, max_files=max_files))
    with ingestion_lock('enlil', FILE_LOCKS['enlil']) as acquired:
        report['enlil'] = (dict(expired=_enlil(now, config.solar_wind_events.retention_days),
            **remove_unreferenced('argus-private', 'observations/ENLIL/', now=now, max_files=max_files))
            if acquired else dict(busy=True, expired=0, files_deleted=0, failed=0))
    report['donki'] = cleanup_donki(now, config.solar_wind_events.retention_days, max_files=max_files)
    return report
