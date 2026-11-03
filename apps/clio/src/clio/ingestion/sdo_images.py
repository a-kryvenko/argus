"""Schedule SDO collection and persist prepared provider observations."""
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import logging

import requests
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy.exc import SQLAlchemyError

from clio.db.locks import ingestion_lock, FILE_LOCKS
from clio.providers import sdo as provider
from clio.observations import sdo
from clio.observations.sdo import COLLECTED_CHANNELS, hourly_slots

logger = logging.getLogger(__name__)
_warmup_cursor = None


def collect_one(slot, channel, *, clock):
    try:
        now = clock()
        if channel not in COLLECTED_CHANNELS:
            return 'unavailable'
        if slot not in hourly_slots(now):
            return 'expired'
        if sdo.scientific_complete(slot, channel):
            return 'existing'
        image, metadata, original = provider.load_observation(channel, slot, now=now)
        now = clock()
        if slot not in hourly_slots(now):
            return 'expired'
        metadata = dict(metadata, version=sdo.VERSION, slot_at=slot.isoformat(), channel=channel,
                        available_at=now.isoformat())
        sdo.publish_scientific(image, metadata, original)
        return 'saved'
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return 'unavailable'
        logger.warning('SDO download failed: %s %s: %s', slot, channel, exc)
    except (requests.RequestException, ValueError, OSError, BotoCoreError, ClientError, SQLAlchemyError) as exc:
        logger.warning('SDO observation failed: %s %s: %s', slot, channel, exc)
    return 'failed'


def cycle(mode, *, batch_hours=6, clock=lambda: datetime.now(UTC), full=False, start=None, end=None, now=None):
    global _warmup_cursor
    if mode not in ('live', 'warmup'):
        raise ValueError('Unknown SDO mode')
    with ingestion_lock('sdo-' + mode, FILE_LOCKS['sdo-' + mode]) as acquired:
        if not acquired:
            return {'busy': True}
        slots = hourly_slots(now if now is not None else clock())
        if mode == 'live':
            slots = slots[:4]
        elif not full:
            slots = slots[4:]
            if _warmup_cursor is not None:
                slots = [s for s in slots if s < _warmup_cursor] + [s for s in slots if s >= _warmup_cursor]
            slots = slots[:batch_hours]
        if start is not None:
            slots = [slot for slot in slots if start <= slot < end]
        report = dict(saved=0, existing=0, unavailable=0, expired=0, failed=0)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for slot in slots:
                for result in pool.map(lambda channel: collect_one(slot, channel, clock=clock), COLLECTED_CHANNELS):
                    report[result] += 1
                if mode == 'warmup' and not full:
                    _warmup_cursor = slot
        logger.info('SDO %s: %s', mode, report)
        return report
