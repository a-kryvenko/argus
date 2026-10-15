"""Session-owned monitoring; clients cannot supply another owner's identity."""
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from common.schemas.drag_monitor import DragProfileInput, DragProfile, DragRun, DragMonitorItem, DragOperatorChart, DragEventPage, DragEvent
from app.dashboard_auth import current_user, check_origin
from app.db.models.dashboard import User
from app.schemas.response import success_response
from app.services.drag_monitor_client import request

router = APIRouter(prefix='/dashboard/drag-monitor', tags=['dashboard'])


@router.get('')
def profiles(user: User = Depends(current_user)):
    return success_response(request('GET', user.id, '', list[DragMonitorItem]))


@router.post('', dependencies=[Depends(check_origin)], status_code=201)
def create(body: DragProfileInput, user: User = Depends(current_user)):
    return success_response(request('POST', user.id, '', DragProfile, body=body))


@router.put('/{profile_id}', dependencies=[Depends(check_origin)])
def update(profile_id: UUID, body: DragProfileInput, revision: int = Query(ge=1), user: User = Depends(current_user)):
    return success_response(request('PUT', user.id, f'/{profile_id}', DragProfile, body=body, params={'revision': revision}))


@router.get('/{profile_id}/history')
def history(profile_id: UUID, limit: int = Query(20, ge=1, le=50), user: User = Depends(current_user)):
    return success_response(request('GET', user.id, f'/{profile_id}/history', list[DragRun], params={'limit': limit}))


@router.post('/{profile_id}/assess', dependencies=[Depends(check_origin)])
def assess(profile_id: UUID, user: User = Depends(current_user)):
    return success_response(request('POST', user.id, f'/{profile_id}/assess', DragRun))


@router.get('/{profile_id}/chart')
def chart(profile_id: UUID, run_id: UUID | None = None, user: User = Depends(current_user)):
    return success_response(request('GET', user.id, f'/{profile_id}/chart', DragOperatorChart,
                                    params={'run_id': str(run_id)} if run_id is not None else None))


@router.get('/{profile_id}/runs/{run_id}')
def run(profile_id: UUID, run_id: UUID, user: User = Depends(current_user)):
    return success_response(request('GET', user.id, f'/{profile_id}/runs/{run_id}', DragRun))


@router.get('/events')
def events(profile_id: UUID | None = None, unacknowledged: bool = False, page: int = Query(1, ge=1, le=100000), user: User = Depends(current_user)):
    params = {'unacknowledged': str(unacknowledged).lower(), 'page': page}
    if profile_id is not None:
        params['profile_id'] = str(profile_id)
    return success_response(request('GET', user.id, '/events', DragEventPage, params=params))


@router.post('/events/{event_id}/ack', dependencies=[Depends(check_origin)])
def acknowledge(event_id: UUID, user: User = Depends(current_user)):
    return success_response(request('POST', user.id, f'/events/{event_id}/ack', DragEvent))
