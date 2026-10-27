"""Authenticated infrastructure for account-owned density advisories."""
from uuid import UUID
from typing import Literal
from sqlalchemy.exc import SQLAlchemyError
from fastapi import APIRouter, HTTPException, Query
from common.schemas.density_advisory import AdvisorySpacecraftInput, AdvisoryProfile, AdvisoryItem, AdvisoryRun, AdvisoryRunSummary, AdvisoryEvent, AdvisoryEventPage
from common.schemas.forecast_release import ForecastRelease
from intelligence import density_monitor as monitor
from intelligence.monitor import ProfileConflict, ProfileNotFound

router=APIRouter(prefix='/internal/v1/density-advisory/{owner_id}')


async def call(operation,*args,**kwargs):
    try:
        from intelligence_core.api import AdvisoryInputError
    except ImportError:
        raise HTTPException(503,'Density advisory backend is unavailable') from None
    try:
        return await operation(*args,**kwargs)
    except AdvisoryInputError as exc:
        raise HTTPException(422,str(exc)) from None
    except ProfileNotFound as exc:
        raise HTTPException(404,str(exc)) from None
    except ProfileConflict as exc:
        raise HTTPException(409,str(exc)) from None
    except (SQLAlchemyError,RuntimeError,ImportError,ValueError,OSError,EOFError):
        raise HTTPException(503,'Density advisory is unavailable') from None


@router.get('',response_model=list[AdvisoryItem])
async def profiles(owner_id:int):
    return await call(monitor.list_profiles,owner_id)


@router.post('',response_model=AdvisoryProfile,status_code=201)
async def create(owner_id:int,body:AdvisorySpacecraftInput):
    return await call(monitor.save_profile,owner_id,body)


@router.put('/{profile_id}',response_model=AdvisoryProfile)
async def update(owner_id:int,profile_id:UUID,body:AdvisorySpacecraftInput,revision:int=Query(ge=1)):
    return await call(monitor.save_profile,owner_id,body,profile_id=profile_id,revision=revision)


@router.delete('/{profile_id}', response_model=dict[str, bool])
async def delete(owner_id:int, profile_id:UUID, revision:int=Query(ge=1)):
    return await call(monitor.delete_profile, owner_id, profile_id, revision=revision)


@router.get('/events',response_model=AdvisoryEventPage)
async def events(owner_id:int,page:int=Query(1,ge=1,le=100000)):
    return await call(monitor.list_events,owner_id,page=page)


@router.post('/events/{event_id}/ack',response_model=AdvisoryEvent)
async def ack(owner_id:int,event_id:UUID):
    return await call(monitor.acknowledge,owner_id,event_id)


@router.post('/{profile_id}/assess',response_model=AdvisoryRun)
async def assess(owner_id:int,profile_id:UUID):
    return await call(monitor.assess_profile,owner_id,profile_id)


@router.get('/{profile_id}',response_model=AdvisoryProfile)
async def profile(owner_id:int,profile_id:UUID):
    return await call(monitor.read_profile,owner_id,profile_id)


@router.get('/{profile_id}/history',response_model=list[AdvisoryRunSummary])
async def history(owner_id:int,profile_id:UUID,limit:int=Query(20,ge=1,le=50)):
    return await call(monitor.history,owner_id,profile_id,limit=limit)


@router.get('/{profile_id}/runs/{run_id}',response_model=AdvisoryRun)
async def run(owner_id:int,profile_id:UUID,run_id:UUID):
    return await call(monitor.read_run,owner_id,profile_id,run_id)


@router.get('/{profile_id}/runs/{run_id}/inputs/{product}', response_model=ForecastRelease)
async def retained_input(owner_id:int, profile_id:UUID, run_id:UUID, product:Literal['atmospheric-density', 'solar-wind-events']):
    return await call(monitor.read_input, owner_id, profile_id, run_id, product)
