"""Collect numerical NOAA runs independently of DONKI availability."""
import asyncio
import logging
from datetime import UTC, datetime
from sqlalchemy import select

from clio.db.models.file_observation import FileObservation
from clio.db.session import get_db_session
from clio.observations.files import store_receipt
from clio.providers import enlil

logger = logging.getLogger(__name__)


async def store(values):
    async with get_db_session() as session:
        session.add(FileObservation(**values))
        await session.commit()


async def run(args):
    try:
        objects = await asyncio.to_thread(enlil.discover, datetime.now(UTC))
        if not objects:
            raise ValueError('No recent compact NOAA ENLIL L1 profiles')
    except (OSError, ValueError, enlil.requests.RequestException) as exc:
        await store(await asyncio.to_thread(store_receipt, 'enlil', 'noaa.enlil', 'discovery', error=type(exc).__name__))
        raise RuntimeError('NOAA ENLIL profile discovery failed') from exc
    failed = []
    for obj in objects:
        metadata = dict(source_url=enlil.BASE_URL+obj['key'], source_key=obj['key'],
            source_modified_at=datetime.fromisoformat(obj['last_modified'].replace('Z', '+00:00')))
        key = obj['key'] + '@' + metadata['source_modified_at'].isoformat()
        async with get_db_session() as session:
            known = await session.scalar(select(FileObservation.id).where(
                FileObservation.kind == 'enlil',
                (FileObservation.source_key == key) | FileObservation.source_key.startswith(key + '#', autoescape=True),
                FileObservation.error.is_(None)).limit(1))
        if known:
            continue
        raw, error = None, None
        try:
            raw = await asyncio.to_thread(enlil.download, metadata['source_url'])
            if len(raw) != obj['size']:
                raise ValueError('ENLIL profile size differs from listing')
            payload = await asyncio.to_thread(enlil.parse_profile, raw)
            metadata['model_created_at'] = payload.pop('model_created_at')
            metadata['payload'] = payload
        except (OSError, ValueError, KeyError, enlil.requests.RequestException) as exc:
            failed.append(obj['key'])
            error = type(exc).__name__
            logger.warning('NOAA ENLIL profile failed: %s', error)
        values = await asyncio.to_thread(store_receipt, 'enlil', 'noaa.enlil', key,
                                        raw=raw, error=error, metadata=metadata)
        await store(values)
    if failed:
        raise RuntimeError(f'NOAA ENLIL failed for {len(failed)} profiles')
    return True
