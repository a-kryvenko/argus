"""Session-owned supplied-trajectory density exposure advisory transport."""
from uuid import UUID
from typing import Literal
from fastapi import APIRouter, Depends, Query
from common.schemas.density_advisory import AdvisorySpacecraftInput, AdvisoryProfile, AdvisoryItem, AdvisoryRun, AdvisoryRunSummary, AdvisoryEvent, AdvisoryEventPage
from common.schemas.forecast_release import ForecastRelease
from api.services.authentication import current_user, check_origin
from api.db.models.dashboard import User
from common.schemas.response import success_response
from api.services.intelligence.profiles import request as forward

router=APIRouter(prefix='/dashboard/density-advisory',tags=['dashboard'])


def request(method,owner_id,suffix,response_type,**kwargs):
    return forward(method,owner_id,suffix,response_type,resource='density-advisory',**kwargs)


@router.get('')
def profiles(user:User=Depends(current_user)):
    return success_response(request('GET',user.id,'',list[AdvisoryItem]))


@router.post('',dependencies=[Depends(check_origin)],status_code=201)
def create(body:AdvisorySpacecraftInput,user:User=Depends(current_user)):
    return success_response(request('POST',user.id,'',AdvisoryProfile,body=body))


@router.put('/{profile_id}',dependencies=[Depends(check_origin)])
def update(profile_id:UUID,body:AdvisorySpacecraftInput,revision:int=Query(ge=1),user:User=Depends(current_user)):
    return success_response(request('PUT',user.id,f'/{profile_id}',AdvisoryProfile,body=body,params={'revision':revision}))


@router.delete('/{profile_id}', dependencies=[Depends(check_origin)])
def delete(profile_id:UUID, revision:int=Query(ge=1), user:User=Depends(current_user)):
    return success_response(request('DELETE', user.id, f'/{profile_id}', dict[str, bool], params={'revision': revision}))


@router.get('/events')
def events(page:int=Query(1,ge=1,le=100000),user:User=Depends(current_user)):
    return success_response(request('GET',user.id,'/events',AdvisoryEventPage,params={'page':page}))


@router.post('/events/{event_id}/ack',dependencies=[Depends(check_origin)])
def ack(event_id:UUID,user:User=Depends(current_user)):
    return success_response(request('POST',user.id,f'/events/{event_id}/ack',AdvisoryEvent))


@router.post('/{profile_id}/assess',dependencies=[Depends(check_origin)])
def assess(profile_id:UUID,user:User=Depends(current_user)):
    return success_response(request('POST',user.id,f'/{profile_id}/assess',AdvisoryRun))


@router.get('/{profile_id}')
def profile(profile_id:UUID,user:User=Depends(current_user)):
    return success_response(request('GET',user.id,f'/{profile_id}',AdvisoryProfile))


@router.get('/{profile_id}/history')
def history(profile_id:UUID,limit:int=Query(20,ge=1,le=50),user:User=Depends(current_user)):
    return success_response(request('GET',user.id,f'/{profile_id}/history',list[AdvisoryRunSummary],params={'limit':limit}))


@router.get('/{profile_id}/runs/{run_id}')
def run(profile_id:UUID,run_id:UUID,user:User=Depends(current_user)):
    return success_response(request('GET',user.id,f'/{profile_id}/runs/{run_id}',AdvisoryRun))


@router.get('/{profile_id}/runs/{run_id}/inputs/{product}')
def retained_input(profile_id:UUID, run_id:UUID, product:Literal['atmospheric-density', 'solar-wind-events'], user:User=Depends(current_user)):
    return success_response(request('GET', user.id,
        f'/{profile_id}/runs/{run_id}/inputs/{product}', ForecastRelease,
        max_response_bytes=64 * 1024 * 1024))
