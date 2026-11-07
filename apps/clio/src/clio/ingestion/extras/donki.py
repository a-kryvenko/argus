"""Download DONKI responses to tmp and save uploaded originals."""
import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from clio.db.locks import ingestion_lock, FILE_LOCKS
from clio.ingestion.utils.file_observations import save
from clio.providers import donki

logger = logging.getLogger(__name__)


def record_poll(endpoint, *, raw, metadata, observed_at=None):
    records = donki.decode(raw)
    with TemporaryDirectory(prefix='clio-donki-') as temporary:
        path = Path(temporary) / 'original.json'
        path.write_bytes(raw)
        return save(path, observation='donki.'+endpoint, observed_at=observed_at or datetime.now(UTC),
                    source_product=endpoint, details=dict(metadata, records=records))['id']


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
