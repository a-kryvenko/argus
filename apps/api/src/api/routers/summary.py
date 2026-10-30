"""Cached homepage data and its generated SVGs."""
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from common.schemas.response import ApiResponse
from api.services.summary import cached_summary, read_image

router = APIRouter(prefix='/summary', tags=['summary'])


@router.get('', response_model=ApiResponse[dict], response_class=JSONResponse)
async def get_summary():
    return Response(await cached_summary(), media_type='application/json', headers={'Cache-Control': 'no-store'})


@router.get('/images/{image_name}', response_class=Response,
            responses={200: {'content': {'image/svg+xml': {}}}})
async def summary_image(image_name: str, request: Request):
    content, ttl = await read_image(image_name)
    headers = {'Cache-Control': f'public, max-age={ttl}, immutable',
               'ETag': f'"{image_name[:-4]}"', 'X-Content-Type-Options': 'nosniff'}
    tags = request.headers.get('if-none-match', '')
    if any(tag.strip().removeprefix('W/') in {headers['ETag'], '*'} for tag in tags.split(',')):
        return Response(status_code=304, headers=headers)
    return Response(content, media_type='image/svg+xml', headers=headers)
