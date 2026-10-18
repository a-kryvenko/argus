"""Authenticated infrastructure for account-owned density advisories."""
from uuid import UUID
from typing import Literal
import psycopg
from fastapi import APIRouter, HTTPException, Query
from common.schemas.density_advisory import AdvisorySpacecraftInput, AdvisoryProfile, AdvisoryItem, AdvisoryRun, AdvisoryRunSummary, AdvisoryEvent, AdvisoryEventPage
from common.schemas.forecast_release import ForecastRelease
from intelligence import density_monitor as monitor
from intelligence.monitor import ProfileConflict, ProfileNotFound

router=APIRouter(prefix='/internal/v1/density-advisory/{owner_id}')


def call(operation,*args,**kwargs):
    try:
        from intelligence_core.api import AdvisoryInputError
    except ImportError:
        raise HTTPException(503,'Density advisory backend is unavailable') from None
    try:
        return operation(*args,**kwargs)
    except AdvisoryInputError as exc:
        raise HTTPException(422,str(exc)) from None
    except ProfileNotFound as exc:
        raise HTTPException(404,str(exc)) from None
    except ProfileConflict as exc:
        raise HTTPException(409,str(exc)) from None
    except (psycopg.Error,RuntimeError,ImportError,ValueError,OSError,EOFError):
        raise HTTPException(503,'Density advisory is unavailable') from None


@router.get('',response_model=list[AdvisoryItem])
def profiles(owner_id:int):
    return call(monitor.list_profiles,owner_id)


@router.post('',response_model=AdvisoryProfile,status_code=201)
def create(owner_id:int,body:AdvisorySpacecraftInput):
    return call(monitor.save_profile,owner_id,body)


@router.put('/{profile_id}',response_model=AdvisoryProfile)
def update(owner_id:int,profile_id:UUID,body:AdvisorySpacecraftInput,revision:int=Query(ge=1)):
    return call(monitor.save_profile,owner_id,body,profile_id=profile_id,revision=revision)


@router.delete('/{profile_id}', response_model=dict[str, bool])
def delete(owner_id:int, profile_id:UUID, revision:int=Query(ge=1)):
    return call(monitor.delete_profile, owner_id, profile_id, revision=revision)


@router.get('/events',response_model=AdvisoryEventPage)
def events(owner_id:int,page:int=Query(1,ge=1,le=100000)):
    return call(monitor.list_events,owner_id,page=page)


@router.post('/events/{event_id}/ack',response_model=AdvisoryEvent)
def ack(owner_id:int,event_id:UUID):
    return call(monitor.acknowledge,owner_id,event_id)


@router.post('/{profile_id}/assess',response_model=AdvisoryRun)
def assess(owner_id:int,profile_id:UUID):
    return call(monitor.assess_profile,owner_id,profile_id)


@router.get('/{profile_id}',response_model=AdvisoryProfile)
def profile(owner_id:int,profile_id:UUID):
    return call(monitor.read_profile,owner_id,profile_id)


@router.get('/{profile_id}/history',response_model=list[AdvisoryRunSummary])
def history(owner_id:int,profile_id:UUID,limit:int=Query(20,ge=1,le=50)):
    return call(monitor.history,owner_id,profile_id,limit=limit)


@router.get('/{profile_id}/runs/{run_id}',response_model=AdvisoryRun)
def run(owner_id:int,profile_id:UUID,run_id:UUID):
    return call(monitor.read_run,owner_id,profile_id,run_id)


@router.get('/{profile_id}/runs/{run_id}/inputs/{product}', response_model=ForecastRelease)
def retained_input(owner_id:int, profile_id:UUID, run_id:UUID, product:Literal['atmospheric-density', 'solar-wind-events']):
    return call(monitor.read_input, owner_id, profile_id, run_id, product)
