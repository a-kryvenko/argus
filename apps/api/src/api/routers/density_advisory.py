"""Session-owned supplied-trajectory density exposure advisory transport."""
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from common.schemas.density_advisory import (
    AdvisoryEvent, AdvisoryEventPage, AdvisoryItem, AdvisoryProfile,
    AdvisoryRun, AdvisoryRunSummary, AdvisorySpacecraftInput,
)
from common.schemas.forecast_release import ForecastRelease
from common.schemas.response import success_response

from api.services.authentication import Principal, check_origin, current_user
from api.services.intelligence.profiles import profile_response, request as forward
from api.services.upstream import IntelligenceClient

router = APIRouter(prefix='/dashboard/density-advisory', tags=['dashboard'])
SPACECRAFT_VALIDATION_ERROR = 'Invalid spacecraft parameters. Check orbital parameters, optional properties and density ratio thresholds.'
VALIDATION_MESSAGES = {
    'High density ratio must exceed review density ratio': 'High density ratio must exceed review density ratio.',
    'Apoapsis, periapsis and inclination are required': 'Apoapsis, periapsis and inclination are required.',
}


async def request(client, method, owner_id, suffix, response_type, **kwargs):
    response = await forward(client, method, f'/internal/v1/density-advisory/{owner_id}{suffix}', validation=True, **kwargs)
    if response.status_code == 422:
        try:
            detail = response.json().get('detail')
        except (ValueError, AttributeError):
            detail = None
        message = VALIDATION_MESSAGES.get(detail, SPACECRAFT_VALIDATION_ERROR) if isinstance(detail, str) else SPACECRAFT_VALIDATION_ERROR
        raise HTTPException(422, message)
    return profile_response(response, response_type)


@router.get('')
async def profiles(client: IntelligenceClient, user: Principal = Depends(current_user)):
    return success_response(await request(client, 'GET', user.id, '', list[AdvisoryItem]))


@router.post('', dependencies=[Depends(check_origin)], status_code=201)
async def create(body: AdvisorySpacecraftInput, client: IntelligenceClient, user: Principal = Depends(current_user)):
    return success_response(await request(client, 'POST', user.id, '', AdvisoryProfile, body=body))


@router.put('/{profile_id}', dependencies=[Depends(check_origin)])
async def update(profile_id: UUID, body: AdvisorySpacecraftInput, client: IntelligenceClient,
                 revision: int = Query(ge=1), user: Principal = Depends(current_user)):
    return success_response(await request(client, 'PUT', user.id, f'/{profile_id}', AdvisoryProfile,
                                          body=body, params={'revision': revision}))


@router.delete('/{profile_id}', dependencies=[Depends(check_origin)])
async def delete(profile_id: UUID, client: IntelligenceClient, revision: int = Query(ge=1),
                 user: Principal = Depends(current_user)):
    return success_response(await request(client, 'DELETE', user.id, f'/{profile_id}', dict[str, bool],
                                          params={'revision': revision}))


@router.get('/events')
async def events(client: IntelligenceClient, page: int = Query(1, ge=1, le=100000), user: Principal = Depends(current_user)):
    return success_response(await request(client, 'GET', user.id, '/events', AdvisoryEventPage, params={'page': page}))


@router.post('/events/{event_id}/ack', dependencies=[Depends(check_origin)])
async def ack(event_id: UUID, client: IntelligenceClient, user: Principal = Depends(current_user)):
    return success_response(await request(client, 'POST', user.id, f'/events/{event_id}/ack', AdvisoryEvent))


@router.post('/{profile_id}/assess', dependencies=[Depends(check_origin)])
async def assess(profile_id: UUID, client: IntelligenceClient, user: Principal = Depends(current_user)):
    return success_response(await request(client, 'POST', user.id, f'/{profile_id}/assess', AdvisoryRun))


@router.get('/{profile_id}')
async def profile(profile_id: UUID, client: IntelligenceClient, user: Principal = Depends(current_user)):
    return success_response(await request(client, 'GET', user.id, f'/{profile_id}', AdvisoryProfile))


@router.get('/{profile_id}/history')
async def history(profile_id: UUID, client: IntelligenceClient, limit: int = Query(20, ge=1, le=50),
                  user: Principal = Depends(current_user)):
    return success_response(await request(client, 'GET', user.id, f'/{profile_id}/history', list[AdvisoryRunSummary],
                                          params={'limit': limit}))


@router.get('/{profile_id}/runs/{run_id}')
async def run(profile_id: UUID, run_id: UUID, client: IntelligenceClient, user: Principal = Depends(current_user)):
    return success_response(await request(client, 'GET', user.id, f'/{profile_id}/runs/{run_id}', AdvisoryRun))


@router.get('/{profile_id}/runs/{run_id}/inputs/{product}')
async def retained_input(profile_id: UUID, run_id: UUID, product: Literal['atmospheric-density', 'solar-wind-events'],
                         client: IntelligenceClient, user: Principal = Depends(current_user)):
    return success_response(await request(client, 'GET', user.id,
                                          f'/{profile_id}/runs/{run_id}/inputs/{product}', ForecastRelease))
