import asyncio
from contextlib import ExitStack
from datetime import datetime, timedelta
import logging

from sqlalchemy import select, delete, exists, or_, and_

from common.data import datamanager
from clio.db import session as db
from clio.db.locks import ingestion_lock, FILE_LOCKS
from clio.db.models.file import File
from clio.db.models.file_observation import FileObservation
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


def expire_files(observation, cutoff, *, max_files):
    counts = dict(expired=0, files_deleted=0, failed=0)
    with db.get_sync_engine().connect() as connection:
        rows = connection.execute(select(FileObservation.id, FileObservation.file_id, FileObservation.details,
            File.bucket, File.key).join(File, File.id == FileObservation.file_id)
            .where(FileObservation.observation == observation, FileObservation.observed_at < cutoff)
            .order_by(FileObservation.observed_at).limit(max_files)).mappings().all()
    for row in rows:
        try:
            for key in [row['key'], *(row['details'] or {}).get('extra_files', [])]:
                datamanager.delete_file(row['bucket'], key)
            with db.get_sync_engine().begin() as connection:
                connection.execute(delete(FileObservation).where(FileObservation.id == row['id']))
                connection.execute(delete(File).where(File.id == row['file_id'],
                    ~exists(select(FileObservation.id).where(FileObservation.file_id == File.id))))
            counts['expired'] += 1
            counts['files_deleted'] += 1
        except Exception:
            counts['failed'] += 1
            logger.exception('Cannot delete expired observation %s', row['id'])
    return counts


def cleanup_files(config, now, *, max_files=MAX_FILE_DELETIONS):
    policies = []
    for observation, policy in config.observations.items():
        definition = OBSERVATIONS[observation]
        if definition.kind != 'file':
            continue
        lanes = {'sdo': ('sdo-live', 'sdo-warmup'), 'sdo_browse': ('sdo-browse',),
                 'gong': ('gong',), 'goes': ('goes',)}[definition.file_kind or observation]
        policies.append((observation, policy.retention_days, lanes))
    from clio.providers.donki import ENDPOINTS
    policies.extend(('donki.'+endpoint, config.solar_wind_events.retention_days, ('donki',)) for endpoint in ENDPOINTS)
    policies.append(('enlil', config.solar_wind_events.retention_days, ('enlil',)))
    report = {}
    for observation, days, lanes in policies:
        with ExitStack() as stack:
            if not all(stack.enter_context(ingestion_lock(lane, FILE_LOCKS[lane])) for lane in lanes):
                report[observation] = dict(busy=True, expired=0, files_deleted=0, failed=0)
                continue
            report[observation] = expire_files(observation, now-timedelta(days=days), max_files=max_files)
    return report
