"""Return the public summary prepared by Prophet."""
from fastapi import APIRouter, HTTPException, Response

from common.schemas.response import ApiResponse, success_response
from api.services.errors import ArtifactNotReadyError
from api.services.forecasts.client import read_summary

router = APIRouter(prefix='/summary', tags=['summary'])


@router.get('', response_model=ApiResponse[dict])
async def get_summary(response: Response):
    response.headers['Cache-Control'] = 'no-store'
    try:
        return success_response(await read_summary())
    except ArtifactNotReadyError:
        raise HTTPException(503, 'Summary is unavailable') from None
