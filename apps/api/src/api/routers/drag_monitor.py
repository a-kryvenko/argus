"""Session-owned monitoring; clients cannot supply another owner's identity."""
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from common.schemas.drag_monitor import (
    DragEvent, DragEventPage, DragMonitorItem, DragOperatorChart, DragProfile, DragProfileInput, DragRun,
)
from common.schemas.response import success_response

from api.services.authentication import Principal, check_origin, current_user
from api.services.intelligence.profiles import profile_response, request as forward
from api.services.upstream import IntelligenceClient

router = APIRouter(prefix='/dashboard/drag-monitor', tags=['dashboard'])


async def request(client, method, owner_id, suffix, response_type, **kwargs):
    response = await forward(client, method, f'/internal/v1/drag-monitor/{owner_id}{suffix}', **kwargs)
    return profile_response(response, response_type)


@router.get('')
async def profiles(client: IntelligenceClient, user: Principal = Depends(current_user)):
    return success_response(await request(client, 'GET', user.id, '', list[DragMonitorItem]))


@router.post('', dependencies=[Depends(check_origin)], status_code=201)
async def create(body: DragProfileInput, client: IntelligenceClient, user: Principal = Depends(current_user)):
    return success_response(await request(client, 'POST', user.id, '', DragProfile, body=body))


@router.put('/{profile_id}', dependencies=[Depends(check_origin)])
async def update(profile_id: UUID, body: DragProfileInput, client: IntelligenceClient,
                 revision: int = Query(ge=1), user: Principal = Depends(current_user)):
    return success_response(await request(client, 'PUT', user.id, f'/{profile_id}', DragProfile,
                                          body=body, params={'revision': revision}))


@router.get('/{profile_id}/history')
async def history(profile_id: UUID, client: IntelligenceClient, limit: int = Query(20, ge=1, le=50), user: Principal = Depends(current_user)):
    return success_response(await request(client, 'GET', user.id, f'/{profile_id}/history', list[DragRun], params={'limit': limit}))


@router.post('/{profile_id}/assess', dependencies=[Depends(check_origin)])
async def assess(profile_id: UUID, client: IntelligenceClient, user: Principal = Depends(current_user)):
    return success_response(await request(client, 'POST', user.id, f'/{profile_id}/assess', DragRun))


@router.get('/{profile_id}/chart')
async def chart(profile_id: UUID, client: IntelligenceClient, run_id: UUID | None = None, user: Principal = Depends(current_user)):
    return success_response(await request(client, 'GET', user.id, f'/{profile_id}/chart', DragOperatorChart,
                                    params={'run_id': str(run_id)} if run_id is not None else None))


@router.get('/{profile_id}/runs/{run_id}')
async def run(profile_id: UUID, run_id: UUID, client: IntelligenceClient, user: Principal = Depends(current_user)):
    return success_response(await request(client, 'GET', user.id, f'/{profile_id}/runs/{run_id}', DragRun))


@router.get('/events')
async def events(client: IntelligenceClient, profile_id: UUID | None = None, unacknowledged: bool = False,
                 page: int = Query(1, ge=1, le=100000), user: Principal = Depends(current_user)):
    params = {'unacknowledged': str(unacknowledged).lower(), 'page': page}
    if profile_id is not None:
        params['profile_id'] = str(profile_id)
    return success_response(await request(client, 'GET', user.id, '/events', DragEventPage, params=params))


@router.post('/events/{event_id}/ack', dependencies=[Depends(check_origin)])
async def acknowledge(event_id: UUID, client: IntelligenceClient, user: Principal = Depends(current_user)):
    return success_response(await request(client, 'POST', user.id, f'/events/{event_id}/ack', DragEvent))
