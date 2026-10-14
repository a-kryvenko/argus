"""Collect numerical NOAA runs independently of DONKI availability."""
import asyncio
import gzip
import hashlib
import logging
from datetime import UTC, datetime
from sqlalchemy import select

from clio.db.models.enlil_profile import EnlilProfileReceipt
from clio.db.session import get_session_factory
from clio.providers import enlil

logger = logging.getLogger(__name__)


async def store(values):
    values['received_at'] = datetime.now(UTC)
    async with get_session_factory()() as session:
        session.add(EnlilProfileReceipt(**values))
        await session.commit()


async def run(args):
    try:
        objects = await asyncio.to_thread(enlil.discover, datetime.now(UTC))
        if not objects:
            raise ValueError('No recent compact NOAA ENLIL L1 profiles')
    except (OSError, ValueError, enlil.requests.RequestException) as exc:
        await store(dict(source_url=enlil.BASE_URL, error=type(exc).__name__))
        raise RuntimeError('NOAA ENLIL profile discovery failed') from exc
    failed = []
    for obj in objects:
        values = dict(source_url=enlil.BASE_URL+obj['key'], source_key=obj['key'],
            source_modified_at=datetime.fromisoformat(obj['last_modified'].replace('Z', '+00:00')))
        async with get_session_factory()() as session:
            known = await session.scalar(select(EnlilProfileReceipt.id).where(
                EnlilProfileReceipt.source_key == obj['key'],
                EnlilProfileReceipt.source_modified_at == values['source_modified_at'],
                EnlilProfileReceipt.error.is_(None)).limit(1))
        if known:
            continue
        try:
            raw = await asyncio.to_thread(enlil.download, values['source_url'])
            if len(raw) != obj['size']:
                raise ValueError('ENLIL profile size differs from listing')
            # Retain exact source bytes even when normalization rejects a file.
            values.update(sha256=hashlib.sha256(raw).hexdigest(), raw_gzip=gzip.compress(raw, mtime=0))
            payload = await asyncio.to_thread(enlil.parse_profile, raw)
            values['model_created_at'] = payload.pop('model_created_at')
            values['payload'] = payload
        except (OSError, ValueError, KeyError, enlil.requests.RequestException) as exc:
            failed.append(obj['key'])
            values['error'] = type(exc).__name__
            logger.warning('NOAA ENLIL profile failed: %s', type(exc).__name__)
        await store(values)
    if failed:
        raise RuntimeError(f'NOAA ENLIL failed for {len(failed)} profiles')
    return True
