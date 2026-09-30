"""Bounded hourly SDO collection with independent live, warmup and retention lanes."""
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import fcntl
import logging
from pathlib import Path

import requests

from clio.config import load_observation_config
from clio.providers.sdo_images import download_observation, observation_url, prepare_observation
from common.sdo_images import (OBSERVED_CHANNELS, archive_root, hourly_slots,
                               image_path, prune, save_image, utc)

logger = logging.getLogger(__name__)


def collect_one(root, slot, channel, *, clock):
    if image_path(root, slot, channel).exists():
        return 'existing'
    try:
        image, metadata = download_observation(observation_url(channel, slot), channel,
                                               slot=slot, now=clock())
        image, metadata = prepare_observation(image, metadata)
        now = clock()
        if slot not in hourly_slots(now):
            return 'expired'
        return 'saved' if save_image(root, slot, channel, image, metadata, now=now) else 'existing'
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return 'unavailable'
        logger.warning('SDO download failed: %s %s: %s', slot, channel, exc)
    except (requests.RequestException, ValueError, OSError) as exc:
        logger.warning('SDO observation failed: %s %s: %s', slot, channel, exc)
    return 'failed'


def cycle(root, mode, *, batch_hours=6, clock=lambda: datetime.now(UTC)):
    """A persistent per-lane lock prevents duplicate workers; publication has its own lock."""
    if mode not in ('live', 'warmup', 'cleanup'):
        raise ValueError('Unknown SDO mode')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / f'.{mode}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'busy': True}
        if mode == 'cleanup':
            report = {'removed': prune(root, now=clock())}
        else:
            slots = hourly_slots(clock())
            cursor = root / '.warmup-cursor'
            if mode == 'live':
                slots = slots[:4]
            else:
                slots = slots[4:]
                if cursor.exists():
                    previous = utc(cursor.read_text().strip())
                    # Continue toward older hours, wrapping even across missing source files.
                    older = [s for s in slots if s < previous]
                    slots = older + [s for s in slots if s >= previous]
                slots = slots[:batch_hours]
            report = dict(saved=0, existing=0, unavailable=0, expired=0, failed=0)
            with ThreadPoolExecutor(max_workers=4) as pool:
                for slot in slots:
                    results = pool.map(lambda channel: collect_one(root, slot, channel, clock=clock),
                                       OBSERVED_CHANNELS)
                    for result in results:
                        report[result] += 1
                    if mode == 'warmup':
                        temporary = root / '.warmup-cursor.part'
                        temporary.write_text(slot.isoformat())
                        temporary.replace(cursor)
        logger.info('SDO %s: %s', mode, report)
        return report


async def run(args):
    policy = load_observation_config().sdo_images
    if not policy.enabled:
        return {'disabled': True}
    return cycle(archive_root(), args.mode, batch_hours=policy.warmup_hours_per_batch)
