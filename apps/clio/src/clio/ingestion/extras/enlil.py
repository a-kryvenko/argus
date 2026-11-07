"""Download compact NOAA runs to tmp and save uploaded originals."""
import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from clio.db.locks import ingestion_lock, FILE_LOCKS
from clio.ingestion.utils.file_observations import find, save
from clio.providers import enlil

logger = logging.getLogger(__name__)


def download(obj):
    with TemporaryDirectory(prefix='clio-enlil-') as temporary:
        path = Path(temporary) / 'original.nc'
        path.write_bytes(enlil.download(enlil.BASE_URL+obj['key']))
        if path.stat().st_size != obj['size']:
            raise ValueError('ENLIL profile size differs from listing')
        payload = enlil.parse_profile(path.read_bytes())
        created = payload.pop('model_created_at')
        return save(path, observation='enlil', observed_at=created, source_product='noaa.enlil',
            details=dict(source_url=enlil.BASE_URL+obj['key'], source_key=obj['key'],
                         source_modified_at=obj['last_modified'], model_created_at=created.isoformat(), payload=payload))


async def collect(*, now=None):
    with ingestion_lock('enlil', FILE_LOCKS['enlil']) as acquired:
        if not acquired:
            return False
        try:
            objects = await asyncio.to_thread(enlil.discover, now or datetime.now(UTC))
        except (OSError, ValueError, enlil.requests.RequestException) as exc:
            raise RuntimeError('NOAA ENLIL discovery failed') from exc
        if not objects:
            raise RuntimeError('No recent compact NOAA ENLIL profiles')
        stored = await asyncio.to_thread(find, 'enlil')
        known = {(row['details']['source_key'], row['details']['source_modified_at']) for row in stored}
        failed = []
        for obj in objects:
            if (obj['key'], obj['last_modified']) in known:
                continue
            try:
                await asyncio.to_thread(download, obj)
            except (OSError, ValueError, KeyError, enlil.requests.RequestException) as exc:
                failed.append(obj['key'])
                logger.warning('NOAA ENLIL profile failed: %s', type(exc).__name__)
        if failed:
            raise RuntimeError(f'NOAA ENLIL failed for {len(failed)} profiles')
        return True
