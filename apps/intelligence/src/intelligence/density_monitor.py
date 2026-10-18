"""Owned density advisory storage and published-input orchestration only."""
from datetime import UTC, datetime
import io
from uuid import uuid4

import pandas as pd
from psycopg.types.json import Jsonb
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


def read_profile(owner_id,profile_id):
    with db.connect() as conn:
        row=conn.execute('SELECT * FROM intelligence.density_profile WHERE owner_id=%s AND id=%s',(owner_id,profile_id)).fetchone()
    if row is None:
        raise ProfileNotFound('Spacecraft not found')
    return profile_from(row)


def save_profile(owner_id, body, *, profile_id=None, revision=None):
    from intelligence_core.api import validate_advisory_profile
    body = body.model_copy(update={'enabled': True})
    validate_advisory_profile(body, now=datetime.now(UTC))
    with db.connect() as conn:
        with conn.transaction():
            if profile_id is None:
                conn.execute('SELECT pg_advisory_xact_lock(736220, %s)', (owner_id,))
                if conn.execute('SELECT count(*) AS n FROM intelligence.density_profile WHERE owner_id=%s', (owner_id,)).fetchone()['n']>=100:
                    raise ProfileConflict('Maximum 100 density profiles per account')
                row=conn.execute('INSERT INTO intelligence.density_profile(id,owner_id,payload) VALUES (%s,%s,%s) RETURNING *',
                    (uuid4(),owner_id,Jsonb(body.model_dump(mode='json')))).fetchone()
            else:
                row=conn.execute('''UPDATE intelligence.density_profile SET payload=%s,revision=revision+1,updated_at=now()
                    WHERE id=%s AND owner_id=%s AND revision=%s RETURNING *''',
                    (Jsonb(body.model_dump(mode='json')),profile_id,owner_id,revision)).fetchone()
                if row is None:
                    if not conn.execute('SELECT 1 FROM intelligence.density_profile WHERE id=%s AND owner_id=%s',(profile_id,owner_id)).fetchone():
                        raise ProfileNotFound('Spacecraft not found')
                    raise ProfileConflict('Profile changed; reload before saving')
    return profile_from(row)


def latest(conn, profile_id):
    row=conn.execute('SELECT payload FROM intelligence.density_run WHERE profile_id=%s ORDER BY assessed_at DESC,id DESC LIMIT 1',(profile_id,)).fetchone()
    return AdvisoryRun.model_validate(row['payload']) if row else None


def delete_profile(owner_id, profile_id, *, revision):
    with db.connect() as conn:
        with conn.transaction():
            row = conn.execute('''SELECT revision FROM intelligence.density_profile
                WHERE owner_id=%s AND id=%s FOR UPDATE''', (owner_id, profile_id)).fetchone()
            if row is None:
                raise ProfileNotFound('Spacecraft not found')
            if row['revision'] != revision:
                raise ProfileConflict('Profile changed; reload before deleting')
            if not conn.execute('SELECT pg_try_advisory_xact_lock(hashtextextended(%s,1)) AS acquired',
                                (str(profile_id),)).fetchone()['acquired']:
                raise ProfileConflict('Assessment is running')
            conn.execute('DELETE FROM intelligence.density_event WHERE profile_id=%s', (profile_id,))
            conn.execute('DELETE FROM intelligence.density_event_state WHERE profile_id=%s', (profile_id,))
            conn.execute('''DELETE FROM intelligence.density_run_input WHERE run_id IN
                (SELECT id FROM intelligence.density_run WHERE profile_id=%s)''', (profile_id,))
            conn.execute('DELETE FROM intelligence.density_run WHERE profile_id=%s', (profile_id,))
            conn.execute('DELETE FROM intelligence.density_profile WHERE owner_id=%s AND id=%s', (owner_id, profile_id))
    return {'deleted': True}


def list_profiles(owner_id):
    from intelligence_core.api import summarize_density_advisory
    with db.connect() as conn:
        with conn.transaction():
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            profiles=conn.execute('SELECT * FROM intelligence.density_profile WHERE owner_id=%s ORDER BY updated_at DESC,id',(owner_id,)).fetchall()
            return [summarize_density_advisory(profile_from(row),latest(conn,row['id']),now=datetime.now(UTC),max_age_hours=max_age()) for row in profiles]


def history(owner_id,profile_id,*,limit=20):
    with db.connect() as conn:
        if not conn.execute('SELECT 1 FROM intelligence.density_profile WHERE id=%s AND owner_id=%s',(profile_id,owner_id)).fetchone():
            raise ProfileNotFound('Spacecraft not found')
        rows=conn.execute('SELECT id,profile_revision,assessed_at AS computed_at FROM intelligence.density_run WHERE profile_id=%s ORDER BY assessed_at DESC,id DESC LIMIT %s',(profile_id,limit)).fetchall()
    return [AdvisoryRunSummary.model_validate(r) for r in rows]


def read_run(owner_id,profile_id,run_id):
    with db.connect() as conn:
        row=conn.execute('''SELECT r.payload FROM intelligence.density_run r JOIN intelligence.density_profile p ON p.id=r.profile_id
            WHERE p.owner_id=%s AND p.id=%s AND r.id=%s''',(owner_id,profile_id,run_id)).fetchone()
    if not row:
        raise ProfileNotFound('Assessment not found')
    return AdvisoryRun.model_validate(row['payload'])


def retrieve_inputs():
    result={}
    for product in ('atmospheric-density','solar-wind-events'):
        try:
            result[product]=fetch_release(product)
        except Exception:
            result[product]=None
    return result


def read_input(owner_id, profile_id, run_id, product):
    with db.connect() as conn:
        row = conn.execute('''SELECT f.* FROM intelligence.density_profile p
            JOIN intelligence.density_run r ON r.profile_id=p.id
            JOIN intelligence.density_run_input i ON i.run_id=r.id
            JOIN intelligence.forecast_input f ON f.release_id=i.release_id
            WHERE p.owner_id=%s AND p.id=%s AND r.id=%s AND i.product=%s''',
            (owner_id, profile_id, run_id, product)).fetchone()
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


def record_events(conn,profile,run,now):
    from intelligence_core.api import evaluate_advisory_events
    row=conn.execute('SELECT payload FROM intelligence.density_event_state WHERE profile_id=%s',(profile.id,)).fetchone()
    drafts,state=evaluate_advisory_events(profile,run,row['payload'] if row else None,now=now,max_age_hours=max_age())
    for event in drafts:
        conn.execute('''INSERT INTO intelligence.density_event(id,profile_id,run_id,detected_at,payload)
            VALUES (%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING''',
            (event.id,event.profile_id,event.run_id,event.detected_at,Jsonb(event.model_dump(mode='json'))))
    conn.execute('''INSERT INTO intelligence.density_event_state(profile_id,payload) VALUES (%s,%s)
        ON CONFLICT (profile_id) DO UPDATE SET payload=excluded.payload''',(profile.id,Jsonb(state)))


def assess_profile(owner_id,profile_id,*,inputs=None):
    with db.connect() as conn:
        if not conn.execute('SELECT pg_try_advisory_lock(hashtextextended(%s,1)) AS acquired',(str(profile_id),)).fetchone()['acquired']:
            raise ProfileConflict('Assessment is running')
        row=conn.execute('SELECT * FROM intelligence.density_profile WHERE owner_id=%s AND id=%s',(owner_id,profile_id)).fetchone()
        if row is None:
            raise ProfileNotFound('Spacecraft not found')
        profile=profile_from(row)
        inputs=retrieve_inputs() if inputs is None else inputs
        previous=latest(conn,profile_id)
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
                assessment=prepare_assessment(snapshot,inputs,now))
        with conn.transaction():
            current=conn.execute('SELECT revision FROM intelligence.density_profile WHERE id=%s FOR UPDATE',(profile_id,)).fetchone()
            if current['revision']!=profile.revision:
                raise ProfileConflict('Profile changed during assessment')
            if not reuse:
                conn.execute('''INSERT INTO intelligence.density_run(id,profile_id,profile_revision,assessed_at,payload)
                    VALUES (%s,%s,%s,%s,%s)''',(run.id,profile.id,profile.revision,run.assessment.computed_at,Jsonb(run.model_dump(mode='json'))))
                forecast_inputs.retain(conn, run.id, inputs, captured_at=now)
            record_events(conn,profile,run,now)
        return run


def list_events(owner_id,*,page=1):
    with db.connect() as conn:
        with conn.transaction():
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            total=conn.execute('''SELECT count(*) AS n FROM intelligence.density_event e
                JOIN intelligence.density_profile p ON p.id=e.profile_id WHERE p.owner_id=%s''',(owner_id,)).fetchone()['n']
            rows=conn.execute('''SELECT e.payload,e.acknowledged_at FROM intelligence.density_event e
                JOIN intelligence.density_profile p ON p.id=e.profile_id WHERE p.owner_id=%s
                ORDER BY e.detected_at DESC,e.id DESC LIMIT 20 OFFSET %s''',(owner_id,(page-1)*20)).fetchall()
    return AdvisoryEventPage(total=total,items=[AdvisoryEvent.model_validate({**r['payload'],'acknowledged_at':r['acknowledged_at']}) for r in rows])


def acknowledge(owner_id,event_id):
    with db.connect() as conn:
        row=conn.execute('''UPDATE intelligence.density_event e SET acknowledged_at=coalesce(e.acknowledged_at,now())
            FROM intelligence.density_profile p WHERE p.id=e.profile_id AND p.owner_id=%s AND e.id=%s
            RETURNING e.payload,e.acknowledged_at''',(owner_id,event_id)).fetchone()
    if row is None:
        raise ProfileNotFound('Event not found')
    return AdvisoryEvent.model_validate({**row['payload'],'acknowledged_at':row['acknowledged_at']})


def refresh_profiles(*,stop=None):
    with db.connect() as conn:
        rows=conn.execute('SELECT id,owner_id FROM intelligence.density_profile').fetchall()
    if not rows:
        return {'profiles':0,'failed':0}
    inputs=retrieve_inputs()
    failed=0
    for row in rows:
        if stop and stop.is_set():
            break
        try:
            assess_profile(row['owner_id'],row['id'],inputs=inputs)
        except Exception:
            failed+=1
    return {'profiles':len(rows),'failed':failed}
