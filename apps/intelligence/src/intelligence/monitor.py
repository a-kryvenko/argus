"""Intelligence-owned profiles, periodic screening, and immutable history."""
import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

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


async def save_profile(owner_id, body: DragProfileInput, *, profile_id=None, revision=None):
    async with db.get_db_session() as session, session.begin():
        if profile_id is None:
            # Bound the work one account can add to the shared polling service.
            await session.execute(text('SELECT pg_advisory_xact_lock(736219, :owner_id)'), {'owner_id': owner_id})
            count = (await session.execute(text('SELECT count(*) AS n FROM intelligence.drag_profile WHERE owner_id=:owner_id'), {'owner_id': owner_id})).mappings().first()['n']
            if count >= 100:
                raise ProfileConflict('Maximum 100 spacecraft profiles per account')
            row = (await session.execute(text('''INSERT INTO intelligence.drag_profile(id,owner_id,payload)
                VALUES (:id,:owner_id,:payload) RETURNING *''').bindparams(bindparam('payload', type_=JSONB)), {'id': uuid4(), 'owner_id': owner_id, 'payload': body.model_dump(mode='json')})).mappings().first()
        else:
            row = (await session.execute(text('''UPDATE intelligence.drag_profile SET payload=:payload, revision=revision+1, updated_at=now()
                WHERE id=:profile_id AND owner_id=:owner_id AND revision=:revision RETURNING *''').bindparams(bindparam('payload', type_=JSONB)), {'payload': body.model_dump(mode='json'), 'profile_id': profile_id, 'owner_id': owner_id, 'revision': revision})).mappings().first()
            if row is None:
                if not (await session.execute(text('SELECT 1 FROM intelligence.drag_profile WHERE id=:profile_id AND owner_id=:owner_id'), {'profile_id': profile_id, 'owner_id': owner_id})).mappings().first():
                    raise ProfileNotFound('Spacecraft not found')
                raise ProfileConflict('Profile changed; reload before saving')
    return profile_from(row)


async def history(owner_id, profile_id, *, limit=20):
    async with db.get_db_session() as session, session.begin():
        if not (await session.execute(text('SELECT 1 FROM intelligence.drag_profile WHERE id=:profile_id AND owner_id=:owner_id'), {'profile_id': profile_id, 'owner_id': owner_id})).mappings().first():
            raise ProfileNotFound('Spacecraft not found')
        rows = (await session.execute(text('''SELECT payload FROM intelligence.drag_run WHERE profile_id=:profile_id
            ORDER BY assessed_at DESC,id DESC LIMIT :limit'''), {'profile_id': profile_id, 'limit': limit})).mappings().all()
    return [DragRun.model_validate(row['payload']) for row in rows]


async def read_run(owner_id, profile_id, run_id):
    async with db.get_db_session() as session, session.begin():
        row = (await session.execute(text('''SELECT r.payload FROM intelligence.drag_run r
            JOIN intelligence.drag_profile p ON p.id=r.profile_id
            WHERE p.owner_id=:owner_id AND p.id=:profile_id AND r.id=:run_id'''), {'owner_id': owner_id, 'profile_id': profile_id, 'run_id': run_id})).mappings().first()
    if row is None:
        raise ProfileNotFound('Assessment not found')
    return DragRun.model_validate(row['payload'])


async def operator_chart(owner_id, profile_id, *, run_id=None):
    try:
        from intelligence_core.api import prepare_operator_chart
    except ImportError:
        raise RuntimeError('Impact backend is not installed') from None
    max_age = get_config().models_registry['models']['atmospheric_density'].get('max_age_hours', 6)
    async with db.get_db_session() as session, session.begin():
        await session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'))
        row = (await session.execute(text('SELECT * FROM intelligence.drag_profile WHERE id=:profile_id AND owner_id=:owner_id'), {'profile_id': profile_id, 'owner_id': owner_id})).mappings().first()
        if row is None:
            raise ProfileNotFound('Spacecraft not found')
        latest = (await session.execute(text('SELECT id,payload FROM intelligence.drag_run WHERE profile_id=:profile_id ORDER BY assessed_at DESC,id DESC LIMIT 1'), {'profile_id': profile_id})).mappings().first()
        selected = ((await session.execute(text('SELECT id,payload FROM intelligence.drag_run WHERE profile_id=:profile_id AND id=:run_id'), {'profile_id': profile_id, 'run_id': run_id})).mappings().first()
                    if run_id is not None else latest)
        if selected is None:
            raise ProfileNotFound('Assessment not found')
        prior = (await session.execute(text('''SELECT payload FROM intelligence.drag_run WHERE profile_id=:profile_id
            AND assessed_at < (SELECT assessed_at FROM intelligence.drag_run WHERE id=:run_id)
            ORDER BY assessed_at DESC,id DESC LIMIT 50'''), {'profile_id': profile_id, 'run_id': selected['id']})).mappings().all()
    return await asyncio.to_thread(prepare_operator_chart, profile_from(row), DragRun.model_validate(selected['payload']),
        [DragRun.model_validate(item['payload']) for item in prior], latest_run_id=latest['id'],
        now=datetime.now(UTC), max_age_hours=max_age)


async def list_profiles(owner_id):
    try:
        from intelligence_core.api import summarize_monitor, prioritize_monitors
    except ImportError:
        raise RuntimeError('Impact backend is not installed') from None
    max_age = get_config().models_registry['models']['atmospheric_density'].get('max_age_hours', 6)
    async with db.get_db_session() as session, session.begin():
        await session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'))
        profiles = (await session.execute(text('SELECT * FROM intelligence.drag_profile WHERE owner_id=:owner_id ORDER BY updated_at DESC,id'), {'owner_id': owner_id})).mappings().all()
        items = []
        for row in profiles:
            runs = (await session.execute(text('''SELECT payload FROM intelligence.drag_run WHERE profile_id=:profile_id
                ORDER BY assessed_at DESC,id DESC LIMIT 20'''), {'profile_id': row['id']})).mappings().all()
            items.append(summarize_monitor(profile_from(row), [DragRun.model_validate(run['payload']) for run in runs], now=datetime.now(UTC), max_age_hours=max_age))
    return prioritize_monitors(items)


async def assess_profile(owner_id, profile_id, *, release=LATEST, automatic=False):
    try:
        from intelligence_core.api import describe_mode
    except ImportError:
        raise RuntimeError('Impact backend is not installed') from None
    async with db.get_db_session() as session, session.begin():
        if not (await session.execute(text('SELECT pg_try_advisory_xact_lock(hashtextextended(:profile_id, 0)) AS acquired'), {'profile_id': str(profile_id)})).mappings().first()['acquired']:
            raise ProfileConflict('Assessment already running')
        row = (await session.execute(text('SELECT * FROM intelligence.drag_profile WHERE id=:profile_id AND owner_id=:owner_id'), {'profile_id': profile_id, 'owner_id': owner_id})).mappings().first()
        if row is None:
            raise ProfileNotFound('Spacecraft not found')
        profile = profile_from(row)
        if not profile.enabled:
            raise ProfileConflict('Monitoring is paused; enable the profile to assess')
        previous = (await session.execute(text('''SELECT payload FROM intelligence.drag_run WHERE profile_id=:profile_id
            ORDER BY assessed_at DESC,id DESC LIMIT 1'''), {'profile_id': profile_id})).mappings().first()
        previous = DragRun.model_validate(previous['payload']) if previous else None
        # Retry failures at most once per five minutes; successful release/profile pairs are immutable.
        if (automatic and previous and previous.profile_revision == profile.revision
                and any(mode.error for mode in previous.modes)
                and previous.release_id == getattr(release, 'release_id', None)
                and datetime.now(UTC) - previous.assessed_at < timedelta(minutes=5)):
            return previous
        try:
            release = await asyncio.to_thread(fetch_release, 'atmospheric-density') if release is LATEST else release
        except Exception:
            release = None
        if previous and release and previous.profile_revision == profile.revision and previous.release_id == release.release_id and all(mode.assessment for mode in previous.modes):
            return previous
        modes = []
        for mode in profile.modes:
            try:
                if release is None:
                    raise DragNotReadyError()
                assessment = await asyncio.to_thread(assess_drag, profile.request_for(mode), release=release)
                modes.append(describe_mode(mode.name, assessment=assessment))
            except DragDomainError:
                modes.append(describe_mode(mode.name, error='outside_model_domain'))
            except DragNotReadyError:
                modes.append(describe_mode(mode.name, error='unavailable'))
        run = DragRun(id=uuid4(), profile_id=profile.id, profile_revision=profile.revision,
                      profile_snapshot=DragProfileInput.model_validate(profile.model_dump(exclude={'id', 'revision', 'updated_at'})),
                      assessed_at=datetime.now(UTC), release_id=release.release_id if release else None, modes=modes)
        current = (await session.execute(text('SELECT revision FROM intelligence.drag_profile WHERE id=:profile_id FOR UPDATE'), {'profile_id': profile_id})).mappings().first()
        if current['revision'] != profile.revision:
            raise ProfileConflict('Profile changed during assessment; calculate again')
        await session.execute(text('''INSERT INTO intelligence.drag_run(id,profile_id,profile_revision,release_id,assessed_at,payload)
            VALUES (:run_id,:profile_id,:revision,:release_id,:assessed_at,:payload)''').bindparams(bindparam('payload', type_=JSONB)), {'run_id': run.id, 'profile_id': profile.id, 'revision': profile.revision, 'release_id': run.release_id, 'assessed_at': run.assessed_at, 'payload': run.model_dump(mode='json')})
        from intelligence.events import record_events
        await record_events(session, profile, [run], now=run.assessed_at)
        return run


async def refresh_profiles(*, stop=None):
    """Use one published release for the cycle; one failed spacecraft does not block others."""
    async with db.get_db_session() as session, session.begin():
        profiles = (await session.execute(text("SELECT id,owner_id FROM intelligence.drag_profile WHERE payload->>'enabled'='true'"))).mappings().all()
    if not profiles:
        return {'profiles': 0, 'failed': 0}
    try:
        release = await asyncio.to_thread(fetch_release, 'atmospheric-density')
    except Exception:
        # assess_profile records an unavailable run, respecting its retry interval.
        release = None
    failed = 0
    for row in profiles:
        if stop is not None and stop.is_set():
            break
        try:
            await assess_profile(row['owner_id'], row['id'], release=release, automatic=True)
            from intelligence.events import process_events
            await process_events(row['owner_id'], row['id'])
        except Exception:
            failed += 1
    return {'profiles': len(profiles), 'failed': failed}
