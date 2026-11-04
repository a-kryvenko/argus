"""Display-only SDO images; scientific FITS have a separate contract."""
from fastapi import APIRouter, HTTPException, Response
from common.schemas.response import success_response
import os
from urllib.parse import quote

from common.schemas.sdo_browse import CHANNELS, SIZES, SDOBrowseCatalog
from clio.db.models.file import File
from clio.db.models.file_observation import FileObservation
from clio.db import session as db
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

router = APIRouter(prefix='/internal/v1/observations/sdo')


@router.get('')
def catalog(response: Response):
    response.headers['Cache-Control'] = 'no-store'
    try:
        return success_response(_catalog())
    except (OSError, ValueError, KeyError, TypeError, SQLAlchemyError):
        raise HTTPException(503, 'SDO image storage is unavailable') from None


def public_entities(sizes):
    return {'jpg.original', *{f'jpg.{size}' for size in sizes if size != 4096}, *{f'webp.{size}' for size in sizes}}


def _catalog():
    base = os.getenv('S3_PUBLIC_BASE_URL', '').rstrip('/')
    statement = (select(FileObservation.variable, FileObservation.details, File.key)
        .join(File, File.id == FileObservation.file_id)
        .where(FileObservation.kind == 'sdo_browse', FileObservation.entity == 'jpg.original',
               FileObservation.details['complete'].as_boolean().is_(True))
        .distinct(FileObservation.variable)
        .order_by(FileObservation.variable, FileObservation.observed_at.desc(), FileObservation.id.desc()))
    with db.get_sync_engine().connect() as connection:
        rows = connection.execute(statement).mappings().all()
    images = {}
    for row in rows:
        metadata = row['details'] or {}
        channel = metadata.get('channel')
        if channel not in CHANNELS or channel in images or not metadata.get('complete'):
            continue
        if not base.startswith(('https://', 'http://')):
            raise ValueError('Set S3_PUBLIC_BASE_URL to the public bucket origin')
        if row['variable'] != 'aia' + str(int(channel)):
            raise ValueError('SDO image channel does not match its receipt')
        images[channel] = {name: metadata.get(name) for name in ('channel', 'asset_id', 'captured_at', 'source_updated_at')}
        prefix = row['key'].rsplit('/', 1)[0]
        images[channel]['urls'] = {entity: base + '/' + quote(prefix + '/' + '.'.join(reversed(entity.split('.'))), safe='/')
                                  for entity in public_entities(SIZES)}
    return SDOBrowseCatalog(images=[images[channel] for channel in CHANNELS if channel in images])
