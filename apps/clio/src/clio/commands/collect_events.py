"""Record DONKI receipts independently; failed requests preserve prior versions."""
import asyncio
import logging
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from clio.config import load_observation_config
from clio.db.models.file_observation import FileObservation
from clio.db.session import get_db_session
from clio.observations.files import store_receipt
from clio.providers import donki

logger = logging.getLogger(__name__)


async def run(args):
    policy = load_observation_config().solar_wind_events
    now = datetime.now(UTC)
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
        receipt_id = uuid4()
        values = await asyncio.to_thread(store_receipt, 'donki', endpoint, str(receipt_id),
            id=receipt_id, raw=raw, error=error, metadata=dict(request_start=start, request_end=end,
                source_url=donki.BASE_URL + endpoint))
        async with get_db_session() as session:
            session.add(FileObservation(**values))
            await session.commit()
    if failed:
        raise RuntimeError('DONKI collection failed: ' + ', '.join(failed))
    return True
