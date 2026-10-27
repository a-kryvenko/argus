"""Owner identity is supplied only by the authenticated API service."""
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from fastapi import APIRouter, HTTPException, Query
from common.schemas.drag_monitor import DragProfileInput, DragProfile, DragMonitorItem, DragRun, DragOperatorChart, DragEventPage, DragEvent
from intelligence import monitor
from intelligence import events

router = APIRouter(prefix='/internal/v1/drag-monitor/{owner_id}')


async def call(operation, *args, **kwargs):
    try:
        return await operation(*args, **kwargs)
    except monitor.ProfileNotFound as exc:
        raise HTTPException(404, str(exc)) from None
    except monitor.ProfileConflict as exc:
        raise HTTPException(409, str(exc)) from None
    except (SQLAlchemyError, RuntimeError):
        raise HTTPException(503, 'Drag monitoring storage is unavailable') from None


@router.get('', response_model=list[DragMonitorItem])
async def profiles(owner_id: int):
    return await call(monitor.list_profiles, owner_id)


@router.post('', response_model=DragProfile, status_code=201)
async def create(owner_id: int, body: DragProfileInput):
    return await call(monitor.save_profile, owner_id, body)


@router.put('/{profile_id}', response_model=DragProfile)
async def update(owner_id: int, profile_id: UUID, body: DragProfileInput, revision: int = Query(ge=1)):
    return await call(monitor.save_profile, owner_id, body, profile_id=profile_id, revision=revision)


@router.get('/{profile_id}/history', response_model=list[DragRun])
async def history(owner_id: int, profile_id: UUID, limit: int = Query(20, ge=1, le=50)):
    return await call(monitor.history, owner_id, profile_id, limit=limit)


@router.post('/{profile_id}/assess', response_model=DragRun)
async def assess(owner_id: int, profile_id: UUID):
    return await call(monitor.assess_profile, owner_id, profile_id)


@router.get('/{profile_id}/chart', response_model=DragOperatorChart)
async def chart(owner_id: int, profile_id: UUID, run_id: UUID | None = None):
    return await call(monitor.operator_chart, owner_id, profile_id, run_id=run_id)


@router.get('/{profile_id}/runs/{run_id}', response_model=DragRun)
async def run(owner_id: int, profile_id: UUID, run_id: UUID):
    return await call(monitor.read_run, owner_id, profile_id, run_id)


@router.get('/events', response_model=DragEventPage)
async def event_list(owner_id: int, profile_id: UUID | None = None, unacknowledged: bool = False, page: int = Query(1, ge=1, le=100000)):
    return await call(events.list_events, owner_id, profile_id=profile_id, unacknowledged=unacknowledged, page=page)


@router.post('/events/{event_id}/ack', response_model=DragEvent)
async def ack(owner_id: int, event_id: UUID):
    return await call(events.acknowledge, owner_id, event_id)
