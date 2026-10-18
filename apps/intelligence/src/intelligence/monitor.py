"""Intelligence-owned profiles, periodic screening, and immutable history."""
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from psycopg.types.json import Jsonb
from common.config import get_config
from common.schemas.drag_monitor import (
    DragProfileInput, DragProfile, DragRun,
)
from intelligence.db import session as db
from intelligence.cli import fetch_release
from intelligence.drag import assess_drag, DragDomainError, DragNotReadyError


class ProfileNotFound(ValueError):
    pass


class ProfileConflict(ValueError):
    pass


LATEST = object()


def profile_from(row):
    return DragProfile(**row['payload'], id=row['id'], revision=row['revision'], updated_at=row['updated_at'])


def save_profile(owner_id, body: DragProfileInput, *, profile_id=None, revision=None):
    with db.connect() as conn:
        if profile_id is None:
            # Bound the work one account can add to the shared polling service.
            with conn.transaction():
                conn.execute('SELECT pg_advisory_xact_lock(736219, %s)', (owner_id,))
                count = conn.execute('SELECT count(*) AS n FROM intelligence.drag_profile WHERE owner_id=%s', (owner_id,)).fetchone()['n']
                if count >= 100:
                    raise ProfileConflict('Maximum 100 spacecraft profiles per account')
                row = conn.execute('''INSERT INTO intelligence.drag_profile(id,owner_id,payload)
                    VALUES (%s,%s,%s) RETURNING *''', (uuid4(), owner_id, Jsonb(body.model_dump(mode='json')))).fetchone()
        else:
            row = conn.execute('''UPDATE intelligence.drag_profile SET payload=%s, revision=revision+1, updated_at=now()
                WHERE id=%s AND owner_id=%s AND revision=%s RETURNING *''',
                (Jsonb(body.model_dump(mode='json')), profile_id, owner_id, revision)).fetchone()
            if row is None:
                if not conn.execute('SELECT 1 FROM intelligence.drag_profile WHERE id=%s AND owner_id=%s', (profile_id, owner_id)).fetchone():
                    raise ProfileNotFound('Spacecraft not found')
                raise ProfileConflict('Profile changed; reload before saving')
    return profile_from(row)


def history(owner_id, profile_id, *, limit=20):
    with db.connect() as conn:
        if not conn.execute('SELECT 1 FROM intelligence.drag_profile WHERE id=%s AND owner_id=%s', (profile_id, owner_id)).fetchone():
            raise ProfileNotFound('Spacecraft not found')
        rows = conn.execute('''SELECT payload FROM intelligence.drag_run WHERE profile_id=%s
            ORDER BY assessed_at DESC,id DESC LIMIT %s''', (profile_id, limit)).fetchall()
    return [DragRun.model_validate(row['payload']) for row in rows]


def read_run(owner_id, profile_id, run_id):
    with db.connect() as conn:
        row = conn.execute('''SELECT r.payload FROM intelligence.drag_run r
            JOIN intelligence.drag_profile p ON p.id=r.profile_id
            WHERE p.owner_id=%s AND p.id=%s AND r.id=%s''', (owner_id, profile_id, run_id)).fetchone()
    if row is None:
        raise ProfileNotFound('Assessment not found')
    return DragRun.model_validate(row['payload'])


def operator_chart(owner_id, profile_id, *, run_id=None):
    try:
        from intelligence_core.api import prepare_operator_chart
    except ImportError:
        raise RuntimeError('Impact backend is not installed') from None
    max_age = get_config().models_registry['models']['atmospheric_density'].get('max_age_hours', 6)
    with db.connect() as conn:
        with conn.transaction():
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            row = conn.execute('SELECT * FROM intelligence.drag_profile WHERE id=%s AND owner_id=%s', (profile_id, owner_id)).fetchone()
            if row is None:
                raise ProfileNotFound('Spacecraft not found')
            latest = conn.execute('SELECT id,payload FROM intelligence.drag_run WHERE profile_id=%s ORDER BY assessed_at DESC,id DESC LIMIT 1', (profile_id,)).fetchone()
            selected = (conn.execute('SELECT id,payload FROM intelligence.drag_run WHERE profile_id=%s AND id=%s', (profile_id, run_id)).fetchone()
                        if run_id is not None else latest)
            if selected is None:
                raise ProfileNotFound('Assessment not found')
            prior = conn.execute('''SELECT payload FROM intelligence.drag_run WHERE profile_id=%s
                AND assessed_at < (SELECT assessed_at FROM intelligence.drag_run WHERE id=%s)
                ORDER BY assessed_at DESC,id DESC LIMIT 50''', (profile_id, selected['id'])).fetchall()
    return prepare_operator_chart(profile_from(row), DragRun.model_validate(selected['payload']),
        [DragRun.model_validate(item['payload']) for item in prior], latest_run_id=latest['id'],
        now=datetime.now(UTC), max_age_hours=max_age)


def list_profiles(owner_id):
    try:
        from intelligence_core.api import summarize_monitor, prioritize_monitors
    except ImportError:
        raise RuntimeError('Impact backend is not installed') from None
    max_age = get_config().models_registry['models']['atmospheric_density'].get('max_age_hours', 6)
    with db.connect() as conn:
        with conn.transaction():
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            profiles = conn.execute('SELECT * FROM intelligence.drag_profile WHERE owner_id=%s ORDER BY updated_at DESC,id', (owner_id,)).fetchall()
            items = []
            for row in profiles:
                runs = conn.execute('''SELECT payload FROM intelligence.drag_run WHERE profile_id=%s
                    ORDER BY assessed_at DESC,id DESC LIMIT 20''', (row['id'],)).fetchall()
                items.append(summarize_monitor(profile_from(row), [DragRun.model_validate(run['payload']) for run in runs], now=datetime.now(UTC), max_age_hours=max_age))
    return prioritize_monitors(items)


def assess_profile(owner_id, profile_id, *, release=LATEST, automatic=False):
    try:
        from intelligence_core.api import describe_mode
    except ImportError:
        raise RuntimeError('Impact backend is not installed') from None
    with db.connect() as conn:
        if not conn.execute('SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS acquired', (str(profile_id),)).fetchone()['acquired']:
            raise ProfileConflict('Assessment already running')
        row = conn.execute('SELECT * FROM intelligence.drag_profile WHERE id=%s AND owner_id=%s', (profile_id, owner_id)).fetchone()
        if row is None:
            raise ProfileNotFound('Spacecraft not found')
        profile = profile_from(row)
        if not profile.enabled:
            raise ProfileConflict('Monitoring is paused; enable the profile to assess')
        previous = conn.execute('''SELECT payload FROM intelligence.drag_run WHERE profile_id=%s
            ORDER BY assessed_at DESC,id DESC LIMIT 1''', (profile_id,)).fetchone()
        previous = DragRun.model_validate(previous['payload']) if previous else None
        # Retry failures at most once per five minutes; successful release/profile pairs are immutable.
        if (automatic and previous and previous.profile_revision == profile.revision
                and any(mode.error for mode in previous.modes)
                and previous.release_id == getattr(release, 'release_id', None)
                and datetime.now(UTC) - previous.assessed_at < timedelta(minutes=5)):
            return previous
        try:
            release = fetch_release('atmospheric-density') if release is LATEST else release
        except Exception:
            release = None
        if previous and release and previous.profile_revision == profile.revision and previous.release_id == release.release_id and all(mode.assessment for mode in previous.modes):
            return previous
        modes = []
        for mode in profile.modes:
            try:
                if release is None:
                    raise DragNotReadyError()
                assessment = assess_drag(profile.request_for(mode), release=release)
                modes.append(describe_mode(mode.name, assessment=assessment))
            except DragDomainError:
                modes.append(describe_mode(mode.name, error='outside_model_domain'))
            except DragNotReadyError:
                modes.append(describe_mode(mode.name, error='unavailable'))
        run = DragRun(id=uuid4(), profile_id=profile.id, profile_revision=profile.revision,
                      profile_snapshot=DragProfileInput.model_validate(profile.model_dump(exclude={'id', 'revision', 'updated_at'})),
                      assessed_at=datetime.now(UTC), release_id=release.release_id if release else None, modes=modes)
        with conn.transaction():
            current = conn.execute('SELECT revision FROM intelligence.drag_profile WHERE id=%s FOR UPDATE', (profile_id,)).fetchone()
            if current['revision'] != profile.revision:
                raise ProfileConflict('Profile changed during assessment; calculate again')
            conn.execute('''INSERT INTO intelligence.drag_run(id,profile_id,profile_revision,release_id,assessed_at,payload)
                VALUES (%s,%s,%s,%s,%s,%s)''', (run.id, profile.id, profile.revision, run.release_id,
                                             run.assessed_at, Jsonb(run.model_dump(mode='json'))))
            from intelligence.events import record_events
            record_events(conn, profile, [run], now=run.assessed_at)
        return run


def refresh_profiles(*, stop=None):
    """Use one published release for the cycle; one failed spacecraft does not block others."""
    with db.connect() as conn:
        profiles = conn.execute("SELECT id,owner_id FROM intelligence.drag_profile WHERE payload->>'enabled'='true'").fetchall()
    if not profiles:
        return {'profiles': 0, 'failed': 0}
    try:
        release = fetch_release('atmospheric-density')
    except Exception:
        # assess_profile records an unavailable run, respecting its retry interval.
        release = None
    failed = 0
    for row in profiles:
        if stop is not None and stop.is_set():
            break
        try:
            assess_profile(row['owner_id'], row['id'], release=release, automatic=True)
            from intelligence.events import process_events
            process_events(row['owner_id'], row['id'])
        except Exception:
            failed += 1
    return {'profiles': len(profiles), 'failed': failed}
