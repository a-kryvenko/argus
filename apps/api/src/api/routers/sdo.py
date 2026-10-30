"""Public display images, collected and prepared by Clio."""
import os
import re

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from common.schemas.sdo_browse import IMAGE_PATTERN, CACHE_CONTROL
from api.services.sdo import image_client

router = APIRouter(prefix='/sdo', tags=['observations'])


@router.get('/images/{name}')
async def image(name: str, request: Request):
    if not re.fullmatch(IMAGE_PATTERN, name):
        raise HTTPException(404, 'SDO image not found')
    url, token = os.getenv('OBSERVATIONS_URL'), os.getenv('OBSERVATIONS_SERVICE_TOKEN')
    if not url or not token:
        raise HTTPException(503, 'Observation service is not configured')
    client = image_client()
    headers = {'Authorization': f'Bearer {token}'}
    if 'if-none-match' in request.headers:
        headers['If-None-Match'] = request.headers['if-none-match']
    try:
        upstream = await client.send(client.build_request('GET',
            url.rstrip('/') + '/internal/v1/observations/sdo-browse/' + name, headers=headers), stream=True)
        if upstream.status_code not in (200, 304):
            status = upstream.status_code
            await upstream.aclose()
            raise HTTPException(404 if status == 404 else 503,
                                'SDO image not found' if status == 404 else 'SDO images are unavailable')
    except httpx.HTTPError:
        raise HTTPException(503, 'SDO images are unavailable') from None
    result_headers = {'Cache-Control': CACHE_CONTROL, 'X-Content-Type-Options': 'nosniff'}
    if upstream.headers.get('etag'):
        result_headers['ETag'] = upstream.headers['etag']
    if upstream.status_code == 304:
        await upstream.aclose()
        return Response(status_code=304, headers=result_headers)

    async def content():
        try:
            async for block in upstream.aiter_bytes():
                yield block
        finally:
            await upstream.aclose()
    return StreamingResponse(content(), media_type='image/webp' if name.endswith('.webp') else 'image/jpeg',
                             headers=result_headers)
