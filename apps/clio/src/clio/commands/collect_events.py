"""Record every DONKI request independently; failed requests preserve prior versions."""
import asyncio
import gzip
import hashlib
import logging
from datetime import UTC, datetime, timedelta

from clio.config import load_observation_config
from clio.db.models.donki_poll import DonkiPoll
from clio.db.session import get_db_session
from clio.observations.transition import mirror_observation
from clio.providers import donki

logger = logging.getLogger(__name__)


async def run(args):
    policy = load_observation_config().solar_wind_events
    now = datetime.now(UTC)
    start = (now - timedelta(days=policy.lookback_days)).replace(hour=0, minute=0, second=0, microsecond=0)
    end = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    failed = []
    for endpoint in donki.ENDPOINTS:
        values = dict(endpoint=endpoint, request_start=start, request_end=end,
                      source_url=donki.BASE_URL + endpoint)
        try:
            raw, payload = await asyncio.to_thread(donki.fetch, endpoint, start, now)
            values.update(sha256=hashlib.sha256(raw).hexdigest(),
                          raw_gzip=gzip.compress(raw, mtime=0), payload=payload)
        except (OSError, ValueError, donki.requests.RequestException) as exc:
            failed.append(endpoint)
            values['error'] = type(exc).__name__
            logger.warning('DONKI %s failed: %s', endpoint, type(exc).__name__)
        # Actual local receipt, never the historical event or model completion time.
        values['received_at'] = datetime.now(UTC)
        async with get_db_session() as session:
            row = DonkiPoll(**values)
            session.add(row)
            await session.flush()
            await mirror_observation(session, 'donki', id=row.id)
            await session.commit()
    if failed:
        raise RuntimeError('DONKI collection failed: ' + ', '.join(failed))
    return True
