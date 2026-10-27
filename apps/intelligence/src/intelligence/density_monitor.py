"""Owned density advisory storage and published-input orchestration only."""
import asyncio
from datetime import UTC, datetime
import io
from uuid import uuid4

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

from common.config import get_config
from common.density_contract import parse_density_frame
from common.schemas.density_advisory import AdvisoryProfile, AdvisoryProfileInput, AdvisoryRun, AdvisoryRunSummary, AdvisoryEvent, AdvisoryEventPage
from common.schemas.solar_wind_events import SolarWindEventForecast
from intelligence import forecast_inputs
from intelligence.db import session as db
from intelligence.cli import fetch_release
from intelligence.monitor import ProfileConflict, ProfileNotFound


def max_age():
    return get_config().models_registry['models']['atmospheric_density'].get('max_age_hours', 6)


def profile_from(row):
    return AdvisoryProfile(**{**row['payload'], 'enabled': True}, id=row['id'], revision=row['revision'], updated_at=row['updated_at'])


async def read_profile(owner_id,profile_id):
    async with db.get_db_session() as session, session.begin():
        row=(await session.execute(text('SELECT * FROM intelligence.density_profile WHERE owner_id=:owner_id AND id=:profile_id'), {'owner_id': owner_id, 'profile_id': profile_id})).mappings().first()
    if row is None:
        raise ProfileNotFound('Spacecraft not found')
    return profile_from(row)


async def save_profile(owner_id, body, *, profile_id=None, revision=None):
    from intelligence_core.api import validate_advisory_profile
    body = body.model_copy(update={'enabled': True})
    validate_advisory_profile(body, now=datetime.now(UTC))
    async with db.get_db_session() as session, session.begin():
        if profile_id is None:
            await session.execute(text('SELECT pg_advisory_xact_lock(736220, :owner_id)'), {'owner_id': owner_id})
            if (await session.execute(text('SELECT count(*) AS n FROM intelligence.density_profile WHERE owner_id=:owner_id'), {'owner_id': owner_id})).mappings().first()['n']>=100:
                raise ProfileConflict('Maximum 100 density profiles per account')
            row=(await session.execute(text('INSERT INTO intelligence.density_profile(id,owner_id,payload) VALUES (:id,:owner_id,:payload) RETURNING *').bindparams(bindparam('payload', type_=JSONB)), {'id': uuid4(), 'owner_id': owner_id, 'payload': body.model_dump(mode='json')})).mappings().first()
        else:
            row=(await session.execute(text('''UPDATE intelligence.density_profile SET payload=:payload,revision=revision+1,updated_at=now()
                WHERE id=:profile_id AND owner_id=:owner_id AND revision=:revision RETURNING *''').bindparams(bindparam('payload', type_=JSONB)), {'payload': body.model_dump(mode='json'), 'profile_id': profile_id, 'owner_id': owner_id, 'revision': revision})).mappings().first()
            if row is None:
                if not (await session.execute(text('SELECT 1 FROM intelligence.density_profile WHERE id=:profile_id AND owner_id=:owner_id'), {'profile_id': profile_id, 'owner_id': owner_id})).mappings().first():
                    raise ProfileNotFound('Spacecraft not found')
                raise ProfileConflict('Profile changed; reload before saving')
    return profile_from(row)


async def latest(session, profile_id):
    row=(await session.execute(text('SELECT payload FROM intelligence.density_run WHERE profile_id=:profile_id ORDER BY assessed_at DESC,id DESC LIMIT 1'), {'profile_id': profile_id})).mappings().first()
    return AdvisoryRun.model_validate(row['payload']) if row else None


async def delete_profile(owner_id, profile_id, *, revision):
    async with db.get_db_session() as session, session.begin():
        row = (await session.execute(text('''SELECT revision FROM intelligence.density_profile
            WHERE owner_id=:owner_id AND id=:profile_id FOR UPDATE'''), {'owner_id': owner_id, 'profile_id': profile_id})).mappings().first()
        if row is None:
            raise ProfileNotFound('Spacecraft not found')
        if row['revision'] != revision:
            raise ProfileConflict('Profile changed; reload before deleting')
        if not (await session.execute(text('SELECT pg_try_advisory_xact_lock(hashtextextended(:profile_id,1)) AS acquired'), {'profile_id': str(profile_id)})).mappings().first()['acquired']:
            raise ProfileConflict('Assessment is running')
        await session.execute(text('DELETE FROM intelligence.density_event WHERE profile_id=:profile_id'), {'profile_id': profile_id})
        await session.execute(text('DELETE FROM intelligence.density_event_state WHERE profile_id=:profile_id'), {'profile_id': profile_id})
        await session.execute(text('''DELETE FROM intelligence.density_run_input WHERE run_id IN
            (SELECT id FROM intelligence.density_run WHERE profile_id=:profile_id)'''), {'profile_id': profile_id})
        await session.execute(text('DELETE FROM intelligence.density_run WHERE profile_id=:profile_id'), {'profile_id': profile_id})
        await session.execute(text('DELETE FROM intelligence.density_profile WHERE owner_id=:owner_id AND id=:profile_id'), {'owner_id': owner_id, 'profile_id': profile_id})
    return {'deleted': True}


async def list_profiles(owner_id):
    from intelligence_core.api import summarize_density_advisory
    async with db.get_db_session() as session, session.begin():
        await session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'))
        profiles=(await session.execute(text('SELECT * FROM intelligence.density_profile WHERE owner_id=:owner_id ORDER BY updated_at DESC,id'), {'owner_id': owner_id})).mappings().all()
        return [summarize_density_advisory(profile_from(row),(await latest(session,row['id'])),now=datetime.now(UTC),max_age_hours=max_age()) for row in profiles]


async def history(owner_id,profile_id,*,limit=20):
    async with db.get_db_session() as session, session.begin():
        if not (await session.execute(text('SELECT 1 FROM intelligence.density_profile WHERE id=:profile_id AND owner_id=:owner_id'), {'profile_id': profile_id, 'owner_id': owner_id})).mappings().first():
            raise ProfileNotFound('Spacecraft not found')
        rows=(await session.execute(text('SELECT id,profile_revision,assessed_at AS computed_at FROM intelligence.density_run WHERE profile_id=:profile_id ORDER BY assessed_at DESC,id DESC LIMIT :limit'), {'profile_id': profile_id, 'limit': limit})).mappings().all()
    return [AdvisoryRunSummary.model_validate(r) for r in rows]


async def read_run(owner_id,profile_id,run_id):
    async with db.get_db_session() as session, session.begin():
        row=(await session.execute(text('''SELECT r.payload FROM intelligence.density_run r JOIN intelligence.density_profile p ON p.id=r.profile_id
            WHERE p.owner_id=:owner_id AND p.id=:profile_id AND r.id=:run_id'''), {'owner_id': owner_id, 'profile_id': profile_id, 'run_id': run_id})).mappings().first()
    if not row:
        raise ProfileNotFound('Assessment not found')
    return AdvisoryRun.model_validate(row['payload'])


async def retrieve_inputs():
    result={}
    for product in ('atmospheric-density','solar-wind-events'):
        try:
            result[product]=await asyncio.to_thread(fetch_release, product)
        except Exception:
            result[product]=None
    return result


async def read_input(owner_id, profile_id, run_id, product):
    async with db.get_db_session() as session, session.begin():
        row = (await session.execute(text('''SELECT f.* FROM intelligence.density_profile p
            JOIN intelligence.density_run r ON r.profile_id=p.id
            JOIN intelligence.density_run_input i ON i.run_id=r.id
            JOIN intelligence.forecast_input f ON f.release_id=i.release_id
            WHERE p.owner_id=:owner_id AND p.id=:profile_id AND r.id=:run_id AND i.product=:product'''), {'owner_id': owner_id, 'profile_id': profile_id, 'run_id': run_id, 'product': product})).mappings().first()
    if row is None:
        raise ProfileNotFound('Retained forecast input not found')
    return forecast_inputs.decode(row)


def prepare_assessment(body,inputs,now):
    from intelligence_core.api import assess_density_exposure
    kwargs={}
    density_release=inputs.get('atmospheric-density')
    if density_release:
        kwargs.update(density_release_id=density_release.release_id,density_published_at=density_release.published_at,
            density_sha256=density_release.artifacts[0].sha256)
        try:
            # Validate at the pinned issue; current freshness is private policy.
            frame=pd.read_csv(io.StringIO(density_release.artifacts[0].csv_text))
            kwargs['density']=parse_density_frame(frame,now=density_release.issue_time)
        except (ValueError,KeyError,IndexError):
            kwargs['density']=None
    cme_release=inputs.get('solar-wind-events')
    if cme_release:
        kwargs.update(cme_release_id=cme_release.release_id,cme_published_at=cme_release.published_at,
            cme_sha256=cme_release.artifacts[0].sha256)
        try:
            frame=pd.read_csv(io.StringIO(cme_release.artifacts[0].csv_text))
            kwargs['cme']=SolarWindEventForecast.model_validate_json(frame.iloc[0]['report_json'])
        except (ValueError,KeyError,IndexError):
            kwargs['cme']=None
    return assess_density_exposure(body,now=now,max_age_hours=max_age(),**kwargs)


async def record_events(session,profile,run,now):
    from intelligence_core.api import evaluate_advisory_events
    row=(await session.execute(text('SELECT payload FROM intelligence.density_event_state WHERE profile_id=:profile_id'), {'profile_id': profile.id})).mappings().first()
    drafts,state=evaluate_advisory_events(profile,run,row['payload'] if row else None,now=now,max_age_hours=max_age())
    for event in drafts:
        await session.execute(text('''INSERT INTO intelligence.density_event(id,profile_id,run_id,detected_at,payload)
            VALUES (:id,:profile_id,:run_id,:detected_at,:payload) ON CONFLICT (id) DO NOTHING''').bindparams(bindparam('payload', type_=JSONB)), {'id': event.id, 'profile_id': event.profile_id, 'run_id': event.run_id, 'detected_at': event.detected_at, 'payload': event.model_dump(mode='json')})
    await session.execute(text('''INSERT INTO intelligence.density_event_state(profile_id,payload) VALUES (:profile_id,:state)
        ON CONFLICT (profile_id) DO UPDATE SET payload=excluded.payload''').bindparams(bindparam('state', type_=JSONB)), {'profile_id': profile.id, 'state': state})


async def assess_profile(owner_id,profile_id,*,inputs=None):
    async with db.get_db_session() as session, session.begin():
        if not (await session.execute(text('SELECT pg_try_advisory_xact_lock(hashtextextended(:profile_id,1)) AS acquired'), {'profile_id': str(profile_id)})).mappings().first()['acquired']:
            raise ProfileConflict('Assessment is running')
        row=(await session.execute(text('SELECT * FROM intelligence.density_profile WHERE owner_id=:owner_id AND id=:profile_id'), {'owner_id': owner_id, 'profile_id': profile_id})).mappings().first()
        if row is None:
            raise ProfileNotFound('Spacecraft not found')
        profile=profile_from(row)
        inputs=(await retrieve_inputs()) if inputs is None else inputs
        previous=await latest(session,profile_id)
        density_id=getattr(inputs.get('atmospheric-density'),'release_id',None)
        cme_id=getattr(inputs.get('solar-wind-events'),'release_id',None)
        now=datetime.now(UTC)
        from intelligence_core.api import reuse_density_advisory
        reuse=reuse_density_advisory(profile,previous,now=now,density_release_id=density_id,cme_release_id=cme_id)
        if reuse:
            run=previous
        else:
            snapshot=AdvisoryProfileInput.model_validate(profile.model_dump(exclude={'id','revision','updated_at'}))
            run=AdvisoryRun(id=uuid4(),profile_id=profile.id,profile_revision=profile.revision,profile_snapshot=snapshot,
                assessment=await asyncio.to_thread(prepare_assessment, snapshot,inputs,now))
        current=(await session.execute(text('SELECT revision FROM intelligence.density_profile WHERE id=:profile_id FOR UPDATE'), {'profile_id': profile_id})).mappings().first()
        if current['revision']!=profile.revision:
            raise ProfileConflict('Profile changed during assessment')
        if not reuse:
            await session.execute(text('''INSERT INTO intelligence.density_run(id,profile_id,profile_revision,assessed_at,payload)
                VALUES (:run_id,:profile_id,:revision,:assessed_at,:payload)''').bindparams(bindparam('payload', type_=JSONB)), {'run_id': run.id, 'profile_id': profile.id, 'revision': profile.revision, 'assessed_at': run.assessment.computed_at, 'payload': run.model_dump(mode='json')})
            await forecast_inputs.retain(session, run.id, inputs, captured_at=now)
        await record_events(session,profile,run,now)
        return run


async def list_events(owner_id,*,page=1):
    async with db.get_db_session() as session, session.begin():
        await session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'))
        total=(await session.execute(text('''SELECT count(*) AS n FROM intelligence.density_event e
            JOIN intelligence.density_profile p ON p.id=e.profile_id WHERE p.owner_id=:owner_id'''), {'owner_id': owner_id})).mappings().first()['n']
        rows=(await session.execute(text('''SELECT e.payload,e.acknowledged_at FROM intelligence.density_event e
            JOIN intelligence.density_profile p ON p.id=e.profile_id WHERE p.owner_id=:owner_id
            ORDER BY e.detected_at DESC,e.id DESC LIMIT 20 OFFSET :offset'''), {'owner_id': owner_id, 'offset': (page-1)*20})).mappings().all()
    return AdvisoryEventPage(total=total,items=[AdvisoryEvent.model_validate({**r['payload'],'acknowledged_at':r['acknowledged_at']}) for r in rows])


async def acknowledge(owner_id,event_id):
    async with db.get_db_session() as session, session.begin():
        row=(await session.execute(text('''UPDATE intelligence.density_event e SET acknowledged_at=coalesce(e.acknowledged_at,now())
            FROM intelligence.density_profile p WHERE p.id=e.profile_id AND p.owner_id=:owner_id AND e.id=:event_id
            RETURNING e.payload,e.acknowledged_at'''), {'owner_id': owner_id, 'event_id': event_id})).mappings().first()
    if row is None:
        raise ProfileNotFound('Event not found')
    return AdvisoryEvent.model_validate({**row['payload'],'acknowledged_at':row['acknowledged_at']})


async def refresh_profiles(*,stop=None):
    async with db.get_db_session() as session, session.begin():
        rows=(await session.execute(text('SELECT id,owner_id FROM intelligence.density_profile'))).mappings().all()
    if not rows:
        return {'profiles':0,'failed':0}
    inputs=await retrieve_inputs()
    failed=0
    for row in rows:
        if stop and stop.is_set():
            break
        try:
            await assess_profile(row['owner_id'],row['id'],inputs=inputs)
        except Exception:
            failed+=1
    return {'profiles':len(rows),'failed':failed}
