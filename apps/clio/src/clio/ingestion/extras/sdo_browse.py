"""Collect public AIA JPEGs, publish their derivatives and list stored references."""
from datetime import datetime
import hashlib
import io
import logging

from PIL import Image
import requests
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy.exc import SQLAlchemyError

from common.schemas.sdo_browse import SIZES
from clio.db.locks import ingestion_lock, FILE_LOCKS
from common.data import datamanager
from clio.ingestion.utils import file_observations
from clio.ingestion.extras import sdo

logger = logging.getLogger(__name__)


def publish_image(content, metadata, *, source=None):
    metadata = {**metadata, 'complete': False, 'sha256': hashlib.sha256(content).hexdigest()}
    metadata.pop('checked_hour', None)
    timestamp, variable = sdo.utc(metadata['captured_at']), 'aia' + str(int(metadata['channel']))
    with Image.open(io.BytesIO(content)) as image:
        if image.format != 'JPEG' or image.size != (4096, 4096):
            raise ValueError('Expected NASA 4096x4096 JPEG')
        original = image.convert('RGB')
    # Publish the unchanged original first; an interrupted batch never becomes the current image.
    sdo.publish(content, variable, timestamp, 'jpg.original', details=metadata, public=True, content_type='image/jpeg', source=source)
    for size in SIZES:
        resized = original if size == 4096 else original.resize((size, size), Image.Resampling.LANCZOS)
        for format in ('jpg', 'webp'):
            if (size, format) == (4096, 'jpg'):
                continue
            output = io.BytesIO()
            resized.save(output, format='JPEG' if format == 'jpg' else 'WEBP', quality=85 if size <= 512 else 90)
            sdo.publish(output.getvalue(), variable, timestamp, f'{format}.{size}', details=metadata,
                public=True, content_type='image/jpeg' if format == 'jpg' else 'image/webp', source=source)
    file_observations.set_details('sdo_browse', variable, timestamp, {**metadata, 'complete': True})


def snapshot(channel, now, *, source=None):
    variable = 'aia' + str(int(channel))
    hour = now.strftime('%Y%m%d%H')
    rows = file_observations.find(kind='sdo_browse', variable=variable, entity='jpg.original')
    previous = next((row for row in rows if (row['details'] or {}).get('complete')), None)
    pending = next((row for row in rows if not (row['details'] or {}).get('complete')), None)
    if pending:
        # Resume processing an interrupted upload in memory, without a local archive.
        publish_image(datamanager.read_reference(pending['file']), pending['details'], source=source)
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
            return
    captured = now.replace(second=0, microsecond=0)
    publish_image(content, dict(channel=channel, asset_id=f'{hour}_{channel}_{digest}',
        captured_at=captured.isoformat(), source_updated_at=updated.isoformat() if updated else None), source=source)


def collect(policy, *, channel, source, mode, now):
    with ingestion_lock('sdo-browse', FILE_LOCKS['sdo-browse']) as acquired:
        if not acquired:
            return {'status': 'busy', 'failed': 0}
        try:
            snapshot(str(int(channel[3:])).zfill(3), now, source=source)
        except (requests.RequestException, OSError, ValueError, BotoCoreError, ClientError, SQLAlchemyError):
            logger.exception('SDO browse collection failed for %s', channel)
            return {'status': 'partial', 'failed': 1}
        return {'status': 'complete', 'failed': 0}
