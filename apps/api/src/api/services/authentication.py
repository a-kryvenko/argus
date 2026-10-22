"""Server-side dashboard sessions."""
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool
from api.schemas.dashboard import UserCreate
from sqlalchemy.ext.asyncio import AsyncSession
from api.db.session import get_db_session
from api.db.models.dashboard import User, Group, Membership, Session

COOKIE = 'argus_session'


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    key = hashlib.scrypt(password.encode(), salt=salt.encode(), n=2**15, r=8, p=3, maxmem=64*1024*1024)
    return f'scrypt${salt}${key.hex()}'


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, salt, expected = encoded.split('$')
        if algorithm != 'scrypt':
            return False
        key = hashlib.scrypt(password.encode(), salt=salt.encode(), n=2**15, r=8, p=3, maxmem=64*1024*1024)
        return hmac.compare_digest(key.hex(), expected)
    except (ValueError, TypeError):
        return False


# Same expensive verification for unknown users; never a usable account.
DUMMY_HASH = hash_password(secrets.token_urlsafe(32))


def check_origin(request: Request):
    """Require the configured frontend origin for all cookie-authenticated writes."""
    origins = {s.strip().rstrip('/') for s in os.getenv('DASHBOARD_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',') if s.strip()}
    if request.headers.get('origin', '').rstrip('/') not in origins:
        raise HTTPException(403, 'Untrusted request origin')


async def current_user(request: Request, db: AsyncSession = Depends(get_db_session)):
    token = request.cookies.get(COOKIE)
    if not token:
        raise HTTPException(401, 'Please sign in')
    user = await db.scalar(select(User).join(Session, Session.user_id == User.id).where(
        Session.token_hash == digest(token), Session.expires_at > datetime.now(timezone.utc), User.active.is_(True)))
    if user is None:
        raise HTTPException(401, 'Session expired')
    return user


async def users_info(db, users):
    groups = {user.id: [] for user in users}
    rows = await db.execute(select(Membership.user_id, Group).join(Group, Group.name == Membership.group_name)
                            .where(Membership.user_id.in_(groups)))
    for user_id, group in rows:
        groups[user_id].append(group)
    return [{'id': user.id, 'username': user.username, 'active': user.active,
             'groups': [g.name for g in groups[user.id]],
             'permissions': sorted({p for g in groups[user.id] for p in g.permissions})} for user in users]


async def user_info(db, user):
    return (await users_info(db, [user]))[0]


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


def require(permission):
    async def dependency(user: User = Depends(current_user), db: AsyncSession = Depends(get_db_session)):
        if permission not in (await user_info(db, user))['permissions']:
            raise HTTPException(403, 'Access denied')
        return user
    return dependency
