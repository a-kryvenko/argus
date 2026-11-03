"""Expire file observations, then remove unreferenced S3 objects under source locks."""
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
import logging

from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import select, delete, exists, or_, and_

from clio.db.locks import ingestion_lock, FILE_LOCKS
from clio.db.models.file import File
from clio.db.models.file_observation import FileObservation as Receipt
from clio.observations import files, file_storage, sdo

logger = logging.getLogger(__name__)
MAX_FILE_DELETIONS = 1000


def remove_unreferenced(bucket, prefix, *, now, max_files=MAX_FILE_DELETIONS):
    """Caller must hold every collection lock for this prefix."""
    counts = dict(files_deleted=0, failed=0)
    referenced = exists(select(Receipt.id).where(Receipt.file_id == File.id))
    with files.engine().connect() as connection:
        unused = connection.execute(select(File.__table__).where(File.bucket == bucket,
            File.key.startswith(prefix, autoescape=True), ~referenced,
            File.timestamp < now-timedelta(minutes=15))
            .order_by(File.timestamp, File.id).limit(max_files)).mappings().all()
    for reference in unused:
        try:
            file_storage.delete(reference)
        except (BotoCoreError, ClientError, OSError):
            counts['failed'] += 1
            logger.exception('Cannot remove unused object %s/%s', reference['bucket'], reference['key'])
            continue
        # Leave the registry entry on S3 failure, so the next cycle can retry.
        with files.engine().begin() as connection:
            connection.execute(delete(File).where(File.id == reference['id'], ~referenced))
        counts['files_deleted'] += 1
    return counts


def _expire(kind, condition):
    with files.engine().begin() as connection:
        return connection.execute(delete(Receipt).where(Receipt.kind == kind, condition)).rowcount


def _browse(now):
    with files.engine().connect() as connection:
        complete = connection.execute(select(Receipt.variable, Receipt.observed_at).where(
            Receipt.kind == 'sdo_browse', Receipt.entity == 'jpg.original',
            Receipt.details['complete'].as_boolean().is_(True))
            .order_by(Receipt.observed_at.desc())).all()
    latest = {}
    for variable, time in complete:
        latest.setdefault(variable, time)
    keep = or_(*(and_(Receipt.variable == variable, Receipt.observed_at == time)
                 for variable, time in latest.items()), False)
    # Preserve the latest complete image and this hour's incomplete publication for retries.
    return _expire('sdo_browse', and_(Receipt.observed_at < now.replace(minute=0, second=0, microsecond=0), ~keep))


def _enlil(now, days):
    cutoff = now-timedelta(days=days)
    with files.engine().connect() as connection:
        old = connection.execute(select(Receipt.__table__).where(
            Receipt.kind == 'enlil', Receipt.received_at < cutoff)).mappings().all()
    def utc(value):
        value = datetime.fromisoformat(value) if isinstance(value, str) else value
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    keep, versions = [], []
    if old:
        keep.append(max(old, key=lambda row: (utc(row['received_at']), row['id']))['id'])
    for row in old:
        if row['error'] is not None:
            continue
        metadata = row['details'] or {}
        try:
            created, modified = utc(metadata['model_created_at']), utc(metadata['source_modified_at'])
        except (KeyError, TypeError, ValueError):
            keep.append(row['id'])  # Unknown metadata must not cause loss of the only successful profile.
            continue
        if created <= cutoff and modified <= cutoff:
            versions.append((created, utc(row['received_at']), row['id']))
        else:
            keep.append(row['id'])
    if versions:
        keep.append(max(versions)[2])
    return _expire('enlil', and_(Receipt.received_at < cutoff, Receipt.id.not_in(keep)))


def cleanup(config, now, *, max_files=MAX_FILE_DELETIONS):
    from clio.observations.donki import cleanup as cleanup_donki
    report = {}
    for kind, lanes, bucket, prefix in (
        ('sdo', ('sdo-live', 'sdo-warmup'), file_storage.PRIVATE_BUCKET, 'images/SDO/'),
        ('sdo_browse', ('sdo-browse',), file_storage.PUBLIC_BUCKET, 'images/AIA/'),
        ('gong', ('gong',), file_storage.PRIVATE_BUCKET, 'observations/GONG/'),
        ('goes', ('goes',), file_storage.PRIVATE_BUCKET, 'observations/GOES/'),
        ('enlil', ('enlil',), file_storage.PRIVATE_BUCKET, 'observations/ENLIL/'),
    ):
        with ExitStack() as stack:
            if not all(stack.enter_context(ingestion_lock(lane, FILE_LOCKS[lane])) for lane in lanes):
                report[kind] = dict(busy=True, expired=0, files_deleted=0, failed=0)
                continue
            expired = 0
            if kind == 'sdo':
                lower = sdo.hourly_slots(now)[-1]
                expired = _expire(kind, or_(Receipt.observed_at < lower,
                                           Receipt.variable.not_in(sdo.COLLECTED_CHANNELS)))
            elif kind == 'sdo_browse':
                expired = _browse(now)
            elif kind == 'enlil':
                expired = _enlil(now, config.solar_wind_events.retention_days)
            elif kind in config.observations:
                cutoff = (now-timedelta(days=config.observations[kind].backfill_days)).replace(
                    minute=0, second=0, microsecond=0)
                if kind == 'goes':
                    cutoff = cutoff.replace(hour=0)
                expired = _expire(kind, or_(Receipt.observed_at < cutoff,
                    and_(Receipt.observed_at.is_(None), Receipt.received_at < cutoff)))
            report[kind] = dict(expired=expired, **remove_unreferenced(bucket, prefix, now=now, max_files=max_files))
    report['donki'] = cleanup_donki(now, config.solar_wind_events.retention_days, max_files=max_files)
    logger.info('File retention: %s', report)
    return report
