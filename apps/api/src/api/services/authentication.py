"""Dashboard credentials, sessions and permissions."""
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import hmac
import os
import secrets

from anyio import CapacityLimiter, to_thread
from fastapi import Depends, HTTPException, Request
from sqlalchemy import case, delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models.dashboard import Group, LoginAttempt, Membership, Session, User
from api.db.session import get_request_db_session

COOKIE = 'argus_session'
SESSION_LIFETIME = timedelta(hours=12)
LOGIN_LIMIT = 10
LOGIN_WINDOW_MINUTES = 15
LOGIN_WINDOW = timedelta(minutes=LOGIN_WINDOW_MINUTES)
_PASSWORD_WORKERS = CapacityLimiter(2)

# Unknown users still pay the same verification cost.
DUMMY_HASH = f"scrypt${'0' * 32}${'0' * 128}"


@dataclass(frozen=True, slots=True)
class Principal:
    id: int
    username: str
    active: bool


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _scrypt(password: str, salt: str) -> str:
    return hashlib.scrypt(password.encode(), salt=salt.encode(), n=2**15, r=8, p=3, maxmem=64*1024*1024).hex()


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    return f'scrypt${salt}${_scrypt(password, salt)}'


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, salt, expected = encoded.split('$')
        return algorithm == 'scrypt' and hmac.compare_digest(_scrypt(password, salt), expected)
    except (ValueError, TypeError):
        return False


async def password_work(function, *args):
    return await to_thread.run_sync(function, *args, limiter=_PASSWORD_WORKERS)


def dashboard_origins() -> frozenset[str]:
    configured = os.getenv('DASHBOARD_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000')
    return frozenset(value.strip().rstrip('/') for value in configured.split(',') if value.strip().rstrip('/'))


def check_origin(request: Request):
    origins = getattr(request.app.state, 'dashboard_origins', None)
    if origins is None:
        origins = dashboard_origins()
    if request.headers.get('origin', '').rstrip('/') not in origins:
        raise HTTPException(403, 'Untrusted request origin')


async def reserve_login_attempt(db: AsyncSession, username: str, now: datetime) -> tuple[int, datetime]:
    window = now.replace(minute=now.minute // LOGIN_WINDOW_MINUTES * LOGIN_WINDOW_MINUTES, second=0, microsecond=0)
    stmt = insert(LoginAttempt).values(key=digest(f'user:{username}'), window=window, count=1)
    stmt = stmt.on_conflict_do_update(index_elements=[LoginAttempt.key], set_={
        'window': func.greatest(LoginAttempt.window, stmt.excluded.window),
        'count': case(
            (LoginAttempt.window < stmt.excluded.window, 1),
            (LoginAttempt.count < LOGIN_LIMIT + 1, LoginAttempt.count + 1),
            else_=LOGIN_LIMIT + 1,
        ),
    })
    count, slot = (await db.execute(stmt.returning(LoginAttempt.count, LoginAttempt.window))).one()
    stale = (select(LoginAttempt.key).where(LoginAttempt.window < window - LOGIN_WINDOW)
             .order_by(LoginAttempt.window).limit(100).with_for_update(skip_locked=True))
    await db.execute(delete(LoginAttempt).where(LoginAttempt.key.in_(stale)).execution_options(synchronize_session=False))
    return count, slot


async def prune_expired_sessions(db: AsyncSession, now: datetime):
    expired = (select(Session.token_hash).where(Session.expires_at <= now)
               .order_by(Session.expires_at).limit(100).with_for_update(skip_locked=True))
    await db.execute(delete(Session).where(Session.token_hash.in_(expired)).execution_options(synchronize_session=False))


async def session_principal(db: AsyncSession, token: str | None, *, lock_user: bool = False) -> Principal:
    if not token:
        raise HTTPException(401, 'Please sign in')
    query = select(User.id, User.username, User.active).join(Session, Session.user_id == User.id).where(
        Session.token_hash == digest(token), Session.expires_at > datetime.now(UTC), User.active.is_(True))
    if lock_user:
        query = query.with_for_update(of=User)
    row = (await db.execute(query)).one_or_none()
    if row is None:
        raise HTTPException(401, 'Session expired')
    return Principal(*row)


async def current_user(request: Request, db: AsyncSession = Depends(get_request_db_session)) -> Principal:
    try:
        return await session_principal(db, request.cookies.get(COOKIE))
    finally:
        await db.rollback()


async def check_permission(db: AsyncSession, user: Principal, permission: str):
    rows = await db.scalars(select(Group.permissions).join(Membership, Group.name == Membership.group_name)
                            .where(Membership.user_id == user.id))
    if not any(permission in permissions for permissions in rows.all()):
        raise HTTPException(403, 'Access denied')


def require(permission: str):
    async def dependency(
            user: Principal = Depends(current_user), db: AsyncSession = Depends(get_request_db_session)) -> Principal:
        try:
            await check_permission(db, user, permission)
            return user
        finally:
            await db.rollback()
    return dependency
