"""Owner identity is supplied only by the authenticated API service."""
from uuid import UUID

import psycopg
from fastapi import APIRouter, HTTPException, Query
from common.schemas.drag_monitor import DragProfileInput, DragProfile, DragMonitorItem, DragRun, DragOperatorChart, DragEventPage, DragEvent
from argus_intelligence import monitor
from argus_intelligence import events

router = APIRouter(prefix='/internal/v1/drag-monitor/{owner_id}')


def call(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except monitor.ProfileNotFound as exc:
        raise HTTPException(404, str(exc)) from None
    except monitor.ProfileConflict as exc:
        raise HTTPException(409, str(exc)) from None
    except (psycopg.Error, RuntimeError):
        raise HTTPException(503, 'Drag monitoring storage is unavailable') from None


@router.get('', response_model=list[DragMonitorItem])
def profiles(owner_id: int):
    return call(monitor.list_profiles, owner_id)


@router.post('', response_model=DragProfile, status_code=201)
def create(owner_id: int, body: DragProfileInput):
    return call(monitor.save_profile, owner_id, body)


@router.put('/{profile_id}', response_model=DragProfile)
def update(owner_id: int, profile_id: UUID, body: DragProfileInput, revision: int = Query(ge=1)):
    return call(monitor.save_profile, owner_id, body, profile_id=profile_id, revision=revision)


@router.get('/{profile_id}/history', response_model=list[DragRun])
def history(owner_id: int, profile_id: UUID, limit: int = Query(20, ge=1, le=50)):
    return call(monitor.history, owner_id, profile_id, limit=limit)


@router.post('/{profile_id}/assess', response_model=DragRun)
def assess(owner_id: int, profile_id: UUID):
    return call(monitor.assess_profile, owner_id, profile_id)


@router.get('/{profile_id}/chart', response_model=DragOperatorChart)
def chart(owner_id: int, profile_id: UUID, run_id: UUID | None = None):
    return call(monitor.operator_chart, owner_id, profile_id, run_id=run_id)


@router.get('/{profile_id}/runs/{run_id}', response_model=DragRun)
def run(owner_id: int, profile_id: UUID, run_id: UUID):
    return call(monitor.read_run, owner_id, profile_id, run_id)


@router.get('/events', response_model=DragEventPage)
def event_list(owner_id: int, profile_id: UUID | None = None, unacknowledged: bool = False, page: int = Query(1, ge=1, le=100000)):
    return call(events.list_events, owner_id, profile_id=profile_id, unacknowledged=unacknowledged, page=page)


@router.post('/events/{event_id}/ack', response_model=DragEvent)
def ack(owner_id: int, event_id: UUID):
    return call(events.acknowledge, owner_id, event_id)
