"""Download validated scientific originals to tmp, upload and store their references."""
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy.exc import SQLAlchemyError

from common.schemas.sdo_images import OBSERVED_CHANNELS
from clio.db.locks import ingestion_lock, FILE_LOCKS
from clio.ingestion.utils.file_observations import find, save
from clio.ingestion.utils.products import FILE_OBSERVATIONS

logger = logging.getLogger(__name__)


def utc(value):
    value = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError('Expected timezone-aware timestamp')
    return value.astimezone(UTC)


def observation_slots(now, *, days, resolution_minutes):
    step = timedelta(minutes=resolution_minutes)
    current = utc(now)
    slot = datetime.fromtimestamp((int(current.timestamp()) // int(step.total_seconds())) * int(step.total_seconds()), UTC)
    lower = current - timedelta(days=days)
    slots = []
    while slot >= lower:
        slots.append(slot)
        slot -= step
    return slots


def collect_one(slot, channel, *, clock, source=None):
    try:
        if channel not in OBSERVED_CHANNELS:
            return 'unavailable'
        observation = FILE_OBSERVATIONS['sdo', channel]
        if find(observation, start=slot-timedelta(minutes=2), end=slot+timedelta(minutes=2, seconds=1)):
            return 'existing'
        from clio.providers import sdo as provider
        url = provider.observation_url(channel, slot)
        with TemporaryDirectory(prefix='clio-sdo-') as temporary:
            path = Path(temporary) / 'original.fits'
            provider.download_file(url, path)
            _, metadata = provider.read_native_fits(path, channel, slot=slot, now=clock())
            save(path, observation=observation, observed_at=utc(metadata['observed_at']),
                source_product=source or ('aia.nrt_' + channel[3:] if channel.startswith('aia') else 'hmi.nrt_m'),
                details=dict(source_url=url, channel=channel))
        return 'saved'
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return 'unavailable'
        logger.warning('SDO download failed: %s %s: %s', slot, channel, exc)
    except (requests.RequestException, ValueError, OSError, BotoCoreError, ClientError, SQLAlchemyError) as exc:
        logger.warning('SDO observation failed: %s %s: %s', slot, channel, exc)
    return 'failed'


def collect(policy, *, channel, source, mode, now):
    lane = 'sdo-live' if mode == 'live' else 'sdo-warmup'
    with ingestion_lock(lane, FILE_LOCKS[lane]) as acquired:
        if not acquired:
            return {'status': 'busy', 'failed': 0}
        resolution = policy.live_resolution_minutes if mode == 'live' else policy.resolution_minutes
        slots = observation_slots(now, days=policy.retention_days, resolution_minutes=resolution)
        horizon = timedelta(days=policy.backfill_days) if mode == 'backfill' else timedelta(minutes=policy.freshness_minutes)
        slots = [slot for slot in slots if slot >= now - horizon]
        report = dict(saved=0, existing=0, unavailable=0, failed=0)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for state in pool.map(lambda slot: collect_one(slot, channel, clock=lambda: datetime.now(UTC), source=source), slots):
                report[state] += 1
        report['status'] = 'partial' if report['failed'] or (mode == 'backfill' and report['unavailable']) else 'complete'
        return report
