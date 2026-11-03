"""Record DONKI receipts independently; failed requests preserve prior versions."""
import asyncio
import logging
from datetime import UTC, datetime, timedelta
from clio.observations.donki import record_poll, lock
from clio.providers import donki

logger = logging.getLogger(__name__)


async def collect(policy, *, now=None):
    with lock() as acquired:
        if not acquired:
            return False
        return await _collect(policy, now=now)


async def _collect(policy, *, now=None):
    now = now or datetime.now(UTC)
    start = (now - timedelta(days=policy.lookback_days)).replace(hour=0, minute=0, second=0, microsecond=0)
    end = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    failed = []
    for endpoint in donki.ENDPOINTS:
        raw, error = None, None
        try:
            raw, _ = await asyncio.to_thread(donki.fetch, endpoint, start, now)
        except (OSError, ValueError, donki.requests.RequestException) as exc:
            failed.append(endpoint)
            error = type(exc).__name__
            logger.warning('DONKI %s failed: %s', endpoint, error)
        await asyncio.to_thread(record_poll, endpoint,
            raw=raw, error=error, metadata=dict(request_start=start, request_end=end,
                source_url=donki.BASE_URL + endpoint))
    if failed:
        raise RuntimeError('DONKI collection failed: ' + ', '.join(failed))
    return True
