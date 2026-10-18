"""Persistence and delivery of private-core event evaluations."""
from datetime import UTC, datetime

from psycopg.types.json import Jsonb
from common.config import get_config
from common.schemas.drag_monitor import DragRun, DragEvent, DragEventPage
from intelligence.db import session as db
from intelligence.monitor import ProfileNotFound, profile_from


def record_events(conn, profile, runs, *, now):
    """Caller holds the profile row lock and commits drafts/state with its write."""
    from intelligence_core.api import evaluate_drag_events
    row = conn.execute('SELECT payload FROM intelligence.drag_event_state WHERE profile_id=%s FOR UPDATE', (profile.id,)).fetchone()
    max_age = get_config().models_registry['models']['atmospheric_density'].get('max_age_hours', 6)
    result = evaluate_drag_events(profile, runs, row['payload'] if row else None, now=now, max_age_hours=max_age)
    for event in result.events:
        conn.execute('''INSERT INTO intelligence.drag_event(id,profile_id,run_id,detected_at,payload)
            VALUES (%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING''',
            (event.id, profile.id, event.run_id, event.detected_at, Jsonb(event.model_dump(mode='json'))))
    conn.execute('''INSERT INTO intelligence.drag_event_state(profile_id,payload) VALUES (%s,%s)
        ON CONFLICT (profile_id) DO UPDATE SET payload=EXCLUDED.payload,updated_at=now()''',
        (profile.id, Jsonb(result.state)))


def process_events(owner_id, profile_id):
    """Clock-driven checks also run when no new forecast release is published."""
    with db.connect() as conn:
        with conn.transaction():
            row = conn.execute('SELECT * FROM intelligence.drag_profile WHERE id=%s AND owner_id=%s FOR UPDATE', (profile_id, owner_id)).fetchone()
            if row is None:
                raise ProfileNotFound('Spacecraft not found')
            latest = conn.execute('SELECT payload FROM intelligence.drag_run WHERE profile_id=%s ORDER BY assessed_at DESC,id DESC LIMIT 1', (profile_id,)).fetchone()
            record_events(conn, profile_from(row), [DragRun.model_validate(latest['payload'])] if latest else [], now=datetime.now(UTC))


def event_from(row):
    return DragEvent(**row['payload'], acknowledged_at=row['acknowledged_at'], acknowledged_by=row['acknowledged_by'])


def list_events(owner_id, *, profile_id=None, unacknowledged=False, page=1):
    with db.connect() as conn:
        with conn.transaction():
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            if profile_id is not None and not conn.execute('SELECT 1 FROM intelligence.drag_profile WHERE id=%s AND owner_id=%s', (profile_id, owner_id)).fetchone():
                raise ProfileNotFound('Spacecraft not found')
            parameters = (owner_id, profile_id, profile_id, unacknowledged)
            where = '''FROM intelligence.drag_event e JOIN intelligence.drag_profile p ON p.id=e.profile_id
                WHERE p.owner_id=%s AND (%s::uuid IS NULL OR e.profile_id=%s)
                AND (NOT %s OR e.acknowledged_at IS NULL)'''
            counts = conn.execute('SELECT count(*) AS total,count(*) FILTER (WHERE e.acknowledged_at IS NULL) AS unacknowledged '+where, parameters).fetchone()
            rows = conn.execute('SELECT e.* '+where+' ORDER BY e.detected_at DESC,e.id DESC LIMIT 20 OFFSET %s', (*parameters, (page-1)*20)).fetchall()
    return DragEventPage(items=[event_from(row) for row in rows], **counts)


def acknowledge(owner_id, event_id):
    with db.connect() as conn:
        row = conn.execute('''UPDATE intelligence.drag_event e
            SET acknowledged_at=coalesce(e.acknowledged_at,now()),acknowledged_by=coalesce(e.acknowledged_by,%s)
            FROM intelligence.drag_profile p WHERE e.profile_id=p.id AND p.owner_id=%s AND e.id=%s
            RETURNING e.*''', (owner_id, owner_id, event_id)).fetchone()
        if row is None:
            raise ProfileNotFound('Event not found')
    return event_from(row)
