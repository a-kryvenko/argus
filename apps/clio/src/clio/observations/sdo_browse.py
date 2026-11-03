"""Collect public AIA JPEGs, publish their derivatives and list stored references."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import io
import logging
import os
from urllib.parse import quote

from PIL import Image
import requests
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy.exc import SQLAlchemyError

from common.schemas.sdo_browse import CHANNELS, SIZES, SDOBrowseCatalog
from clio.db.locks import ingestion_lock
from clio.observations import file_storage, file_observations, sdo

logger = logging.getLogger(__name__)


def public_entities(sizes):
    return {'jpg.original', *{f'jpg.{size}' for size in sizes if size != 4096}, *{f'webp.{size}' for size in sizes}}


def catalog():
    base = os.getenv('S3_PUBLIC_BASE_URL', '').rstrip('/')
    rows = file_observations.find(kind='sdo_browse', entity='jpg.original')
    images = {}
    for row in rows:
        metadata = row['details'] or {}
        channel = metadata.get('channel')
        if channel not in CHANNELS or channel in images or not metadata.get('complete'):
            continue
        if not base.startswith(('https://', 'http://')):
            raise ValueError('Set S3_PUBLIC_BASE_URL to the public bucket origin')
        timestamp = sdo.utc(metadata['captured_at'])
        images[channel] = {name: metadata.get(name) for name in ('channel', 'asset_id', 'captured_at', 'source_updated_at')}
        images[channel]['urls'] = {entity: base + '/' + quote(sdo.object_key(row['variable'], timestamp, entity, public=True), safe='/')
                                  for entity in public_entities(SIZES)}
    return SDOBrowseCatalog(images=[images[channel] for channel in CHANNELS if channel in images])


def publish_image(content, metadata):
    metadata = {**metadata, 'complete': False}
    timestamp, variable = sdo.utc(metadata['captured_at']), 'aia' + str(int(metadata['channel']))
    with Image.open(io.BytesIO(content)) as source:
        if source.format != 'JPEG' or source.size != (4096, 4096):
            raise ValueError('Expected NASA 4096x4096 JPEG')
        original = source.convert('RGB')
    # Publish the unchanged original first; an interrupted batch never becomes the current image.
    sdo.publish(content, variable, timestamp, 'jpg.original', details=metadata, public=True, content_type='image/jpeg')
    for size in SIZES:
        resized = original if size == 4096 else original.resize((size, size), Image.Resampling.LANCZOS)
        for format in ('jpg', 'webp'):
            if (size, format) == (4096, 'jpg'):
                continue
            output = io.BytesIO()
            resized.save(output, format='JPEG' if format == 'jpg' else 'WEBP', quality=85 if size <= 512 else 90)
            sdo.publish(output.getvalue(), variable, timestamp, f'{format}.{size}', details=metadata,
                public=True, content_type='image/jpeg' if format == 'jpg' else 'image/webp')
    file_observations.set_details('sdo_browse', variable, timestamp, {**metadata, 'complete': True})


def snapshot(channel, now):
    variable = 'aia' + str(int(channel))
    hour = now.strftime('%Y%m%d%H')
    rows = file_observations.find(kind='sdo_browse', variable=variable, entity='jpg.original')
    previous = next((row for row in rows if (row['details'] or {}).get('complete')), None)
    if previous and previous['details'].get('checked_hour') == hour:
        return
    pending = next((row for row in rows if not (row['details'] or {}).get('complete')
                    and (row['details'] or {}).get('checked_hour') == hour), None)
    if pending:
        # Resume processing an interrupted upload in memory, without a local archive.
        publish_image(file_storage.read(pending['file']), pending['details'])
        return
    from clio.providers.sdo import download_browse
    content, updated = download_browse(channel)
    digest = hashlib.sha256(b'sdo-browse-v1\0' + content).hexdigest()
    if previous:
        metadata = previous['details']
        unchanged = metadata['asset_id'].endswith('_' + digest)
        if updated and metadata.get('source_updated_at'):
            unchanged |= updated <= sdo.utc(metadata['source_updated_at'])
        if unchanged:
            file_observations.set_details('sdo_browse', variable, previous['timestamp'], {**metadata, 'checked_hour': hour})
            return
    captured = now.replace(second=0, microsecond=0)
    publish_image(content, dict(channel=channel, asset_id=f'{hour}_{channel}_{digest}',
        captured_at=captured.isoformat(), source_updated_at=updated.isoformat() if updated else None, checked_hour=hour))


def collect(now):
    with ingestion_lock('sdo-browse', 730210) as acquired:
        if not acquired:
            return {'status': 'busy', 'failed': []}
        def fetch(channel):
            try:
                snapshot(channel, now)
            except (requests.RequestException, OSError, ValueError, BotoCoreError, ClientError, SQLAlchemyError):
                logger.exception('SDO browse collection failed for AIA %s', channel)
                return channel
            return None
        with ThreadPoolExecutor(max_workers=3) as pool:
            failed = [channel for channel in pool.map(fetch, CHANNELS) if channel is not None]
        return {'status': 'partial' if failed else 'complete', 'failed': failed}
