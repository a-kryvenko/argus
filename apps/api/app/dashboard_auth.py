"""Server-side dashboard sessions."""
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db_session
from app.db.models.dashboard import User, Group, Membership, Session

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


async def user_info(db, user):
    groups = list((await db.scalars(select(Group).join(Membership, Membership.group_name == Group.name)
        .where(Membership.user_id == user.id))).all())
    return {'id': user.id, 'username': user.username, 'active': user.active,
            'groups': [g.name for g in groups], 'permissions': sorted({p for g in groups for p in g.permissions})}


def require(permission):
    async def dependency(user: User = Depends(current_user), db: AsyncSession = Depends(get_db_session)):
        if permission not in (await user_info(db, user))['permissions']:
            raise HTTPException(403, 'Access denied')
        return user
    return dependency
