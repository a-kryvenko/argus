"""Persistence and delivery of private-core event evaluations."""
from datetime import UTC, datetime

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

from common.config import get_config
from common.schemas.drag_monitor import DragRun, DragEvent, DragEventPage
from intelligence.db import session as db
from intelligence.monitor import ProfileNotFound, profile_from


async def record_events(session, profile, runs, *, now):
    """Caller holds the profile row lock and commits drafts/state with its write."""
    from intelligence_core.api import evaluate_drag_events
    row = (await session.execute(text('SELECT payload FROM drag_event_state WHERE profile_id=:profile_id FOR UPDATE'), {'profile_id': profile.id})).mappings().first()
    max_age = get_config().models_registry['models']['atmospheric_density'].get('max_age_hours', 6)
    result = evaluate_drag_events(profile, runs, row['payload'] if row else None, now=now, max_age_hours=max_age)
    for event in result.events:
        await session.execute(text('''INSERT INTO drag_event(id,profile_id,run_id,detected_at,payload)
            VALUES (:id,:profile_id,:run_id,:detected_at,:payload) ON CONFLICT (id) DO NOTHING''').bindparams(bindparam('payload', type_=JSONB)), {'id': event.id, 'profile_id': profile.id, 'run_id': event.run_id, 'detected_at': event.detected_at, 'payload': event.model_dump(mode='json')})
    await session.execute(text('''INSERT INTO drag_event_state(profile_id,payload) VALUES (:profile_id,:state)
        ON CONFLICT (profile_id) DO UPDATE SET payload=EXCLUDED.payload,updated_at=now()''').bindparams(bindparam('state', type_=JSONB)), {'profile_id': profile.id, 'state': result.state})


async def process_events(owner_id, profile_id):
    """Clock-driven checks also run when no new forecast release is published."""
    async with db.get_db_session() as session, session.begin():
        row = (await session.execute(text('SELECT * FROM drag_profile WHERE id=:profile_id AND owner_id=:owner_id FOR UPDATE'), {'profile_id': profile_id, 'owner_id': owner_id})).mappings().first()
        if row is None:
            raise ProfileNotFound('Spacecraft not found')
        latest = (await session.execute(text('SELECT payload FROM drag_run WHERE profile_id=:profile_id ORDER BY assessed_at DESC,id DESC LIMIT 1'), {'profile_id': profile_id})).mappings().first()
        await record_events(session, profile_from(row), [DragRun.model_validate(latest['payload'])] if latest else [], now=datetime.now(UTC))


def event_from(row):
    return DragEvent(**row['payload'], acknowledged_at=row['acknowledged_at'], acknowledged_by=row['acknowledged_by'])


async def list_events(owner_id, *, profile_id=None, unacknowledged=False, page=1):
    async with db.get_db_session() as session, session.begin():
        await session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'))
        if profile_id is not None and not (await session.execute(text('SELECT 1 FROM drag_profile WHERE id=:profile_id AND owner_id=:owner_id'), {'profile_id': profile_id, 'owner_id': owner_id})).mappings().first():
            raise ProfileNotFound('Spacecraft not found')
        parameters = {'owner_id': owner_id, 'profile_id': profile_id, 'unacknowledged': unacknowledged}
        where = """FROM drag_event e JOIN drag_profile p ON p.id=e.profile_id
            WHERE p.owner_id=:owner_id AND (CAST(:profile_id AS uuid) IS NULL OR e.profile_id=:profile_id)
            AND (NOT :unacknowledged OR e.acknowledged_at IS NULL)"""
        counts = (await session.execute(text('SELECT count(*) AS total,count(*) FILTER (WHERE e.acknowledged_at IS NULL) AS unacknowledged ' + where), parameters)).mappings().first()
        rows = (await session.execute(text('SELECT e.* ' + where + ' ORDER BY e.detected_at DESC,e.id DESC LIMIT 20 OFFSET :offset'),
                                      {**parameters, 'offset': (page-1)*20})).mappings().all()
    return DragEventPage(items=[event_from(row) for row in rows], **counts)


async def acknowledge(owner_id, event_id):
    async with db.get_db_session() as session, session.begin():
        row = (await session.execute(text('''UPDATE drag_event e
            SET acknowledged_at=coalesce(e.acknowledged_at,now()),acknowledged_by=coalesce(e.acknowledged_by,:owner_id)
            FROM drag_profile p WHERE e.profile_id=p.id AND p.owner_id=:owner_id AND e.id=:event_id
            RETURNING e.*'''), {'owner_id': owner_id, 'event_id': event_id})).mappings().first()
        if row is None:
            raise ProfileNotFound('Event not found')
    return event_from(row)
