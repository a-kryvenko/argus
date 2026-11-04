"""Collect numerical NOAA runs independently of DONKI availability."""
import asyncio
import logging
from datetime import UTC, datetime
from sqlalchemy import select

from clio.db.models.file_observation import FileObservation
from clio.db.session import get_db_session
from clio.ingestion.utils.archive import store_receipt
from clio.providers import enlil

logger = logging.getLogger(__name__)


async def store(values):
    async with get_db_session() as session:
        session.add(FileObservation(**values))
        await session.commit()


async def collect(*, now=None):
    from clio.db.locks import ingestion_lock, FILE_LOCKS
    with ingestion_lock('enlil', FILE_LOCKS['enlil']) as acquired:
        if not acquired:
            return False
        return await _collect(now=now)


async def _collect(*, now=None):
    try:
        objects = await asyncio.to_thread(enlil.discover, now or datetime.now(UTC))
        if not objects:
            raise ValueError('No recent compact NOAA ENLIL L1 profiles')
    except (OSError, ValueError, enlil.requests.RequestException) as exc:
        raise RuntimeError('NOAA ENLIL profile discovery failed') from exc
    failed = []
    for obj in objects:
        metadata = dict(source_url=enlil.BASE_URL+obj['key'], source_key=obj['key'],
            source_modified_at=datetime.fromisoformat(obj['last_modified'].replace('Z', '+00:00')))
        key = obj['key'] + '@' + metadata['source_modified_at'].isoformat()
        async with get_db_session() as session:
            known = await session.scalar(select(FileObservation.id).where(
                FileObservation.kind == 'enlil',
                (FileObservation.source_key == key) | FileObservation.source_key.startswith(key + '#', autoescape=True)).limit(1))
        if known:
            continue
        try:
            raw = await asyncio.to_thread(enlil.download, metadata['source_url'])
            if len(raw) != obj['size']:
                raise ValueError('ENLIL profile size differs from listing')
            payload = await asyncio.to_thread(enlil.parse_profile, raw)
            metadata['model_created_at'] = payload.pop('model_created_at')
            metadata['payload'] = payload
        except (OSError, ValueError, KeyError, enlil.requests.RequestException) as exc:
            failed.append(obj['key'])
            logger.warning('NOAA ENLIL profile failed: %s', type(exc).__name__)
            continue
        values = await asyncio.to_thread(store_receipt, 'enlil', 'noaa.enlil', key,
                                        raw=raw, metadata=metadata)
        await store(values)
    if failed:
        raise RuntimeError(f'NOAA ENLIL failed for {len(failed)} profiles')
    return True
