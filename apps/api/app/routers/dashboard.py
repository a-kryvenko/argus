from datetime import datetime, timedelta, timezone
from typing import Literal
import os
import secrets

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, delete, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.db.session import get_db_session
from app.db.models.dashboard import User, Group, Membership, Session, LoginAttempt, ApiMetric
from app.dashboard_auth import COOKIE, DUMMY_HASH, check_origin, current_user, digest, hash_password, verify_password, require, user_info
from app.schemas.response import success_response
from app.services.observations_client import read_observations

router = APIRouter(prefix='/dashboard', tags=['dashboard'])


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=80, pattern=r'^[a-zA-Z0-9_.@-]+$')
    password: str = Field(min_length=1, max_length=256)

    @field_validator('username')
    @classmethod
    def normalize(cls, value):
        return value.lower()


class UserCreate(Credentials):
    password: str = Field(min_length=12, max_length=256)
    groups: list[str] = Field(default_factory=list, max_length=50)


class UserUpdate(BaseModel):
    active: bool | None = None
    groups: list[str] | None = Field(default=None, max_length=50)
    password: str | None = Field(default=None, min_length=12, max_length=256)


@router.post('/login', dependencies=[Depends(check_origin)])
async def login(body: Credentials, request: Request, response: Response, db: AsyncSession = Depends(get_db_session)):
    now = datetime.now(timezone.utc)
    window = now.replace(minute=(now.minute // 15)*15, second=0, microsecond=0)
    # Shared database counters work across workers; do not trust forwarded IP headers.
    for label, limit in [(f'peer:{request.client.host if request.client else "unknown"}', 100), (f'user:{body.username}', 10)]:
        stmt = insert(LoginAttempt).values(key=digest(label), window=window, count=1)
        stmt = stmt.on_conflict_do_update(index_elements=[LoginAttempt.key], set_={
            'window': window, 'count': func.coalesce(LoginAttempt.count, 0) + 1})
        # Remove expired counters before incrementing in this transaction.
        await db.execute(delete(LoginAttempt).where(LoginAttempt.window < window))
        count = (await db.execute(stmt.returning(LoginAttempt.count))).scalar_one()
        await db.commit()
        if count > limit:
            raise HTTPException(429, 'Too many sign-in attempts. Try again in 15 minutes.')
    user = await db.scalar(select(User).where(User.username == body.username))
    valid = await run_in_threadpool(verify_password, body.password, user.password_hash if user else DUMMY_HASH)
    if not valid or user is None or not user.active:
        raise HTTPException(401, 'Invalid username or password')
    # Recheck under a row lock so a concurrent block/reset cannot leave a valid session.
    await db.refresh(user, with_for_update=True)
    if not user.active or not await run_in_threadpool(verify_password, body.password, user.password_hash):
        raise HTTPException(401, 'Invalid username or password')
    await db.execute(delete(Session).where(Session.expires_at <= now))
    old_token = request.cookies.get(COOKIE)
    if old_token:
        await db.execute(delete(Session).where(Session.token_hash == digest(old_token)))
    token = secrets.token_urlsafe(32)
    db.add(Session(token_hash=digest(token), user_id=user.id, expires_at=now+timedelta(hours=12)))
    await db.commit()
    response.set_cookie(COOKIE, token, httponly=True, secure=os.getenv('DASHBOARD_COOKIE_SECURE', 'true').lower() != 'false',
                        samesite='strict', max_age=43200, path='/')
    return success_response(await user_info(db, user))


@router.post('/logout', dependencies=[Depends(check_origin)])
async def logout(request: Request, response: Response, db: AsyncSession = Depends(get_db_session)):
    await db.execute(delete(Session).where(Session.token_hash == digest(request.cookies.get(COOKIE, ''))))
    await db.commit()
    response.delete_cookie(COOKIE, path='/')
    return success_response({'signed_out': True})


@router.get('/me')
async def me(user: User = Depends(current_user), db: AsyncSession = Depends(get_db_session)):
    return success_response(await user_info(db, user))


async def validate_groups(db, names):
    names = sorted(set(names))
    known = set((await db.scalars(select(Group.name).where(Group.name.in_(names)))).all())
    if set(names) != known:
        raise HTTPException(422, 'Unknown group')
    return names


async def create_user(db, body: UserCreate):
    groups = await validate_groups(db, body.groups)
    user = User(username=body.username, password_hash=await run_in_threadpool(hash_password, body.password), active=True)
    db.add(user)
    try:
        await db.flush()
        db.add_all([Membership(user_id=user.id, group_name=name) for name in groups])
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(409, 'Username already exists')
    return await user_info(db, user)


@router.get('/groups', dependencies=[Depends(require('users.manage'))])
async def groups(db: AsyncSession = Depends(get_db_session)):
    rows = (await db.scalars(select(Group).order_by(Group.name))).all()
    return success_response([{'name': g.name, 'permissions': g.permissions} for g in rows])


@router.get('/users', dependencies=[Depends(require('users.manage'))])
async def users(page: int = Query(1, ge=1, le=100000), db: AsyncSession = Depends(get_db_session)):
    rows = (await db.scalars(select(User).order_by(User.id).offset((page-1)*50).limit(50))).all()
    return success_response({'items': [await user_info(db, u) for u in rows], 'total': await db.scalar(select(func.count()).select_from(User))})


@router.post('/users', dependencies=[Depends(check_origin), Depends(require('users.manage'))])
async def add_user(body: UserCreate, db: AsyncSession = Depends(get_db_session)):
    return success_response(await create_user(db, body))


@router.patch('/users/{user_id}', dependencies=[Depends(check_origin)])
async def update_user(user_id: int, body: UserUpdate, actor: User = Depends(require('users.manage')), db: AsyncSession = Depends(get_db_session)):
    # Serialize membership changes to preserve at least one active administrator.
    await db.scalar(select(Group).where(Group.name == 'admins').with_for_update())
    user = await db.get(User, user_id, with_for_update=True)
    if user is None:
        raise HTTPException(404, 'User not found')
    names = await validate_groups(db, body.groups) if body.groups is not None else None
    if actor.id == user.id and (body.active is False or (names is not None and 'admins' not in names)):
        raise HTTPException(409, 'You cannot block yourself or remove your own admin access')
    existing = (await user_info(db, user))['groups']
    if user.active and 'admins' in existing and (body.active is False or (names is not None and 'admins' not in names)):
        others = await db.scalar(select(func.count()).select_from(User).join(Membership).where(
            User.active.is_(True), User.id != user.id, Membership.group_name == 'admins'))
        if not others:
            raise HTTPException(409, 'At least one active administrator is required')
    if body.active is not None:
        user.active = body.active
    if body.password is not None:
        user.password_hash = await run_in_threadpool(hash_password, body.password)
    if names is not None:
        await db.execute(delete(Membership).where(Membership.user_id == user.id))
        db.add_all([Membership(user_id=user.id, group_name=name) for name in names])
    if body.active is False or body.password is not None or names is not None:
        await db.execute(delete(Session).where(Session.user_id == user.id))
    await db.commit()
    return success_response(await user_info(db, user))


@router.get('/observations', dependencies=[Depends(require('observations.read'))])
async def observations(kind: Literal['raw', 'normalized'] = 'raw', page: int = Query(1, ge=1, le=100000),
    page_size: int = Query(50, ge=1, le=200), order: Literal['asc', 'desc'] = 'desc',
    start: datetime | None = None, end: datetime | None = None,
    metric: str | None = Query(None, max_length=16)):
    return await read_observations('browse', {
        'kind': kind, 'page': page, 'page_size': page_size, 'order': order,
        'start': start, 'end': end, 'metric': metric,
    })


@router.get('/api-stats', dependencies=[Depends(require('api_stats.read'))])
async def api_stats(hours: int = Query(24, ge=1, le=720), db: AsyncSession = Depends(get_db_session)):
    from app.services.api_statistics import summarize
    since = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours-1)
    rows = (await db.scalars(select(ApiMetric).where(ApiMetric.hour >= since))).all()
    return success_response(summarize(rows))


@router.get('/project-monitoring', dependencies=[Depends(require('project_monitoring.read'))])
async def project_monitoring(db: AsyncSession = Depends(get_db_session)):
    from app.db.models.monitoring import MonitorState
    from app.services.project_monitoring import STALE_SECONDS
    state = await db.get(MonitorState, 'project')
    now = datetime.now(timezone.utc)
    if state is None:
        return success_response({'status': 'unknown', 'checked_at': None, 'stale': True,
            'services': [], 'observations': None, 'forecasts': [], 'host': None})
    stale = (now-state.checked_at).total_seconds() > STALE_SECONDS
    return success_response({**state.payload, 'checked_at': state.checked_at,
        'stale': stale, 'status': 'unknown' if stale else state.payload['status']})


@router.get('/project-traffic', dependencies=[Depends(require('project_monitoring.read'))])
async def project_traffic(period: Literal['hour', 'day', 'week'] = 'day', db: AsyncSession = Depends(get_db_session)):
    from app.db.models.monitoring import MonitorState, TrafficMetric
    from app.services.edge_traffic import traffic_summary
    now = datetime.now(timezone.utc)
    resolution = 'minute' if period == 'hour' else 'hour'
    until = now.replace(second=0, microsecond=0, **({'minute': 0} if resolution == 'hour' else {}))
    since = until - (timedelta(minutes=59) if period == 'hour' else timedelta(hours=23 if period == 'day' else 167))
    state = await db.get(MonitorState, 'traffic')
    stale = not state or (now-state.checked_at).total_seconds() > 120
    payload = state.payload if state else {}
    last_event = payload.get('last_event_at')
    if not last_event or (now-datetime.fromisoformat(last_event)).total_seconds() > 120:
        stale = True
    rows = (await db.scalars(select(TrafficMetric).where(TrafficMetric.resolution == resolution,
        TrafficMetric.time >= since, TrafficMetric.time <= until))).all()
    recent_errors = await db.scalar(select(func.coalesce(func.sum(TrafficMetric.count), 0)).where(
        TrafficMetric.resolution == 'minute', TrafficMetric.time >= now.replace(second=0, microsecond=0)-timedelta(minutes=4),
        TrafficMetric.status >= 500))
    return success_response({'recent_errors_5xx': recent_errors,
        'status': 'unknown' if stale else payload.get('status', 'unknown'),
        'checked_at': state.checked_at if state else None, 'stale': bool(stale),
        'since': payload.get('since'), 'resolution': resolution,
        'channels': traffic_summary(rows, since, until, resolution) if state else None})
