"""Prepare public AIA images in tmp and store only completed uploads."""
import hashlib
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
import requests
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy.exc import SQLAlchemyError

from common.schemas.sdo_browse import SIZES
from clio.db.locks import ingestion_lock, FILE_LOCKS
from clio.ingestion.utils.file_observations import find, save

logger = logging.getLogger(__name__)


def snapshot(channel, now, *, source=None):
    from clio.providers.sdo import download_browse_file
    observation = f'aia_{channel}_browse'
    with TemporaryDirectory(prefix='clio-browse-') as temporary:
        root = Path(temporary)
        path = root / 'original.jpg'
        updated = download_browse_file(channel, path)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        previous = find(observation)
        if previous and previous[0]['sha256'] == digest:
            return
        variants = []
        with Image.open(path) as image:
            if image.format != 'JPEG' or image.size != (4096, 4096):
                raise ValueError('Expected NASA 4096x4096 JPEG')
            original = image.convert('RGB')
            for size in SIZES:
                resized = original if size == 4096 else original.resize((size, size), Image.Resampling.LANCZOS)
                for format in ('jpg', 'webp'):
                    if (size, format) == (4096, 'jpg'):
                        continue
                    variant = root / f'{size}.{format}'
                    resized.save(variant, format='JPEG' if format == 'jpg' else 'WEBP', quality=85 if size <= 512 else 90)
                    variants.append(variant)
        captured = now.replace(second=0, microsecond=0)
        save(path, observation=observation, observed_at=captured,
            source_product=source or 'nasa.aia_browse', bucket='argus-public', variants=variants,
            details=dict(channel=channel, captured_at=captured.isoformat(),
                asset_id=f'{captured:%Y%m%d%H}_{channel}_{digest}',
                source_updated_at=updated.isoformat() if updated else None))


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
