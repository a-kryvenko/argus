from datetime import datetime, timedelta, timezone
import os
import secrets

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from api.schemas.dashboard import Credentials, UserCreate, UserUpdate
from sqlalchemy import select, delete, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from api.db.session import get_db_session, get_request_db_session
from api.db.models.dashboard import User, Group, Membership, Session, LoginAttempt
from api.services.authentication import COOKIE, DUMMY_HASH, check_origin, current_user, digest, hash_password, verify_password, require, user_info, users_info, validate_groups, create_user
from common.schemas.response import success_response

router = APIRouter(prefix='/dashboard', tags=['dashboard'])


@router.post('/login', dependencies=[Depends(check_origin)])
async def login(body: Credentials, request: Request, response: Response, db: AsyncSession = Depends(get_request_db_session)):
    now = datetime.now(timezone.utc)
    window = now.replace(minute=(now.minute // 15)*15, second=0, microsecond=0)
    # Shared account counters work across workers and client IPs. The production
    # nginx applies the per-IP limit after resolving the trusted edge proxy;
    # request.client here is the nginx peer, shared by all visitors.
    stmt = insert(LoginAttempt).values(key=digest(f'user:{body.username}'), window=window, count=1)
    stmt = stmt.on_conflict_do_update(index_elements=[LoginAttempt.key], set_={
        'window': window, 'count': func.coalesce(LoginAttempt.count, 0) + 1})
    await db.execute(delete(LoginAttempt).where(LoginAttempt.window < window))
    count = (await db.execute(stmt.returning(LoginAttempt.count))).scalar_one()
    await db.commit()
    if count > 10:
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
async def logout(request: Request, response: Response, db: AsyncSession = Depends(get_request_db_session)):
    await db.execute(delete(Session).where(Session.token_hash == digest(request.cookies.get(COOKIE, ''))))
    await db.commit()
    response.delete_cookie(COOKIE, path='/')
    return success_response({'signed_out': True})


@router.get('/me')
async def me(user: User = Depends(current_user), db: AsyncSession = Depends(get_request_db_session)):
    return success_response(await user_info(db, user))


@router.get('/groups', dependencies=[Depends(require('users.manage'))])
async def groups(db: AsyncSession = Depends(get_request_db_session)):
    rows = (await db.scalars(select(Group).order_by(Group.name))).all()
    return success_response([{'name': g.name, 'permissions': g.permissions} for g in rows])


@router.get('/users', dependencies=[Depends(require('users.manage'))])
async def users(page: int = Query(1, ge=1, le=100000), db: AsyncSession = Depends(get_request_db_session)):
    rows = (await db.scalars(select(User).order_by(User.id).offset((page-1)*50).limit(50))).all()
    return success_response({'items': await users_info(db, rows), 'total': await db.scalar(select(func.count()).select_from(User))})


@router.post('/users', dependencies=[Depends(check_origin), Depends(require('users.manage'))])
async def add_user(body: UserCreate, db: AsyncSession = Depends(get_request_db_session)):
    return success_response(await create_user(db, body))


@router.patch('/users/{user_id}', dependencies=[Depends(check_origin)])
async def update_user(user_id: int, body: UserUpdate, actor: User = Depends(require('users.manage')), db: AsyncSession = Depends(get_request_db_session)):
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


@router.get('/project-monitoring', dependencies=[Depends(require('project_monitoring.read'))])
async def project_monitoring(response: Response):
    from api.services.project_status import read_project_status
    response.headers['Cache-Control'] = 'no-store'
    return success_response(await read_project_status())
