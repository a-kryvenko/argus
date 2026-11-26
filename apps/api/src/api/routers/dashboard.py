from datetime import UTC, datetime
import math
import os
import secrets

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from api.schemas.dashboard import Credentials, UserCreate, UserUpdate
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.session import get_request_db_session
from api.db.models.dashboard import Group, Session, User
from api.services.authentication import (
    COOKIE, DUMMY_HASH, LOGIN_LIMIT, LOGIN_WINDOW, SESSION_LIFETIME, Principal, check_origin, current_user,
    digest, hash_password, password_work, prune_expired_sessions, require, reserve_login_attempt, verify_password,
)
from api.services.dashboard_users import (
    MANAGE_USERS, authorize_user_change, create_user, update_user as change_user, user_info, users_info,
)
from common.schemas.response import success_response
from api.services.upstream import MonitoringClient

router = APIRouter(prefix='/dashboard', tags=['dashboard'])


def cookie_options():
    return dict(httponly=True, secure=os.getenv('DASHBOARD_COOKIE_SECURE', 'true').lower() != 'false',
                samesite='strict', path='/')


@router.post('/login', dependencies=[Depends(check_origin)])
async def login(body: Credentials, request: Request, response: Response, db: AsyncSession = Depends(get_request_db_session)):
    now = datetime.now(UTC)
    # Shared account counters work across workers and client IPs. The production
    # nginx applies the per-IP limit after resolving the trusted edge proxy;
    # request.client here is the nginx peer, shared by all visitors.
    count, window = await reserve_login_attempt(db, body.username, now)
    if count > LOGIN_LIMIT:
        await db.commit()
        retry = max(1, math.ceil((window + LOGIN_WINDOW - datetime.now(UTC)).total_seconds()))
        raise HTTPException(429, 'Too many sign-in attempts. Try again in 15 minutes.', headers={'Retry-After': str(retry)})

    account = (await db.execute(select(User.id, User.password_hash).where(User.username == body.username))).one_or_none()
    await prune_expired_sessions(db, now)
    await db.commit()
    valid = await password_work(verify_password, body.password, account.password_hash if account else DUMMY_HASH)
    if account is None or not valid:
        raise HTTPException(401, 'Invalid username or password')

    # Confirm that the verified credentials are still current while creating
    # the session, keeping the KDF outside the database transaction.
    user = await db.scalar(select(User).where(
        User.id == account.id, User.active.is_(True), User.password_hash == account.password_hash,
    ).with_for_update())
    if user is None:
        raise HTTPException(401, 'Invalid username or password')
    old_token = request.cookies.get(COOKIE)
    if old_token:
        await db.execute(delete(Session).where(Session.token_hash == digest(old_token)))
    token = secrets.token_urlsafe(32)
    db.add(Session(token_hash=digest(token), user_id=user.id, expires_at=datetime.now(UTC) + SESSION_LIFETIME))
    result = await user_info(db, user)
    await db.commit()
    response.set_cookie(COOKIE, token, max_age=int(SESSION_LIFETIME.total_seconds()), **cookie_options())
    return success_response(result)


@router.post('/logout', dependencies=[Depends(check_origin)])
async def logout(request: Request, response: Response, db: AsyncSession = Depends(get_request_db_session)):
    token = request.cookies.get(COOKIE)
    if token:
        await db.execute(delete(Session).where(Session.token_hash == digest(token)))
        await db.commit()
    response.delete_cookie(COOKIE, **cookie_options())
    return success_response({'signed_out': True})


@router.get('/me')
async def me(user: Principal = Depends(current_user), db: AsyncSession = Depends(get_request_db_session)):
    return success_response(await user_info(db, user))


@router.get('/groups', dependencies=[Depends(require(MANAGE_USERS))])
async def groups(db: AsyncSession = Depends(get_request_db_session)):
    rows = (await db.scalars(select(Group).order_by(Group.name))).all()
    return success_response([{'name': g.name, 'permissions': g.permissions} for g in rows])


@router.get('/users', dependencies=[Depends(require(MANAGE_USERS))])
async def users(page: int = Query(1, ge=1, le=100000), db: AsyncSession = Depends(get_request_db_session)):
    rows = (await db.scalars(select(User).order_by(User.id).offset((page-1)*50).limit(50))).all()
    return success_response({'items': await users_info(db, rows), 'total': await db.scalar(select(func.count()).select_from(User))})


@router.post('/users', dependencies=[Depends(check_origin), Depends(require(MANAGE_USERS))])
async def add_user(body: UserCreate, request: Request, db: AsyncSession = Depends(get_request_db_session)):
    password_hash = await password_work(hash_password, body.password)
    await authorize_user_change(db, request.cookies.get(COOKIE))
    return success_response(await create_user(db, body, password_hash=password_hash))


@router.patch('/users/{user_id}', dependencies=[Depends(check_origin), Depends(require(MANAGE_USERS))])
async def update_user(user_id: int, body: UserUpdate, request: Request, db: AsyncSession = Depends(get_request_db_session)):
    password_hash = await password_work(hash_password, body.password) if body.password is not None else None
    actor = await authorize_user_change(db, request.cookies.get(COOKIE))
    return success_response(await change_user(db, actor, user_id, body, password_hash))


@router.get('/project-monitoring', dependencies=[Depends(require('project_monitoring.read'))])
async def project_monitoring(response: Response, client: MonitoringClient):
    from api.services.project_status import read_project_status
    response.headers['Cache-Control'] = 'no-store'
    return success_response(await read_project_status(client))
