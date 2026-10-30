"""Display-only SDO images; scientific FITS have a separate contract."""
from fastapi import APIRouter, HTTPException, Request, Response
import hashlib
from fastapi.responses import FileResponse
from common.schemas.response import success_response
from common.schemas.sdo_browse import CACHE_CONTROL
from clio.observations import sdo_browse

router = APIRouter(prefix='/internal/v1/observations/sdo-browse')


@router.get('')
def catalog():
    return success_response(sdo_browse.catalog())


@router.get('/{name}')
def image(name: str, request: Request):
    try:
        path = sdo_browse.image_path(name)
    except FileNotFoundError:
        raise HTTPException(404, 'SDO image not found') from None
    headers = {'Cache-Control': CACHE_CONTROL, 'X-Content-Type-Options': 'nosniff',
               'ETag': '"'+hashlib.sha256(name.encode()).hexdigest()+'"'}
    if any(tag.strip().removeprefix('W/') in {headers['ETag'], '*'}
           for tag in request.headers.get('if-none-match', '').split(',')):
        return Response(status_code=304, headers=headers)
    return FileResponse(path, media_type='image/webp' if name.endswith('.webp') else 'image/jpeg', headers=headers)
