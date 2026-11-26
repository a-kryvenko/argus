"""Dashboard account changes and their session lifetime effects."""
from fastapi import HTTPException
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models.dashboard import Group, Membership, Session, User
from api.schemas.dashboard import UserCreate, UserUpdate
from api.services.authentication import Principal, check_permission, hash_password, password_work, session_principal

ADMINS = 'admins'
MANAGE_USERS = 'users.manage'


async def authorize_user_change(db: AsyncSession, token: str | None) -> Principal:
    # Membership mutations share this lock. Authorize under it so a queued
    # change cannot outlive an administrator's concurrent removal.
    group = await db.scalar(select(Group.name).where(Group.name == ADMINS).with_for_update())
    if group is None:
        raise HTTPException(503, 'Dashboard user management is unavailable')
    actor = await session_principal(db, token, lock_user=True)
    await check_permission(db, actor, MANAGE_USERS)
    return actor


async def users_info(db: AsyncSession, users):
    if not users:
        return []
    groups = {user.id: [] for user in users}
    rows = await db.execute(
        select(Membership.user_id, Group).join(Group, Group.name == Membership.group_name)
        .where(Membership.user_id.in_(groups)).order_by(Membership.user_id, Group.name)
    )
    for user_id, group in rows:
        groups[user_id].append(group)
    return [
        {'id': user.id, 'username': user.username, 'active': user.active,
         'groups': [group.name for group in groups[user.id]],
         'permissions': sorted({name for group in groups[user.id] for name in group.permissions})}
        for user in users
    ]


async def user_info(db: AsyncSession, user):
    return (await users_info(db, [user]))[0]


async def validate_groups(db: AsyncSession, names: list[str]) -> list[str]:
    names = sorted(set(names))
    if names:
        known = set((await db.scalars(select(Group.name).where(Group.name.in_(names)))).all())
        if set(names) != known:
            raise HTTPException(422, 'Unknown group')
    return names


async def create_user(db: AsyncSession, body: UserCreate, *, password_hash: str | None = None):
    if password_hash is None:
        password_hash = await password_work(hash_password, body.password)
    groups = await validate_groups(db, body.groups)
    statement = insert(User).values(username=body.username, password_hash=password_hash, active=True)
    user = await db.scalar(statement.on_conflict_do_nothing(index_elements=[User.username]).returning(User))
    if user is None:
        raise HTTPException(409, 'Username already exists')
    db.add_all([Membership(user_id=user.id, group_name=name) for name in groups])
    result = await user_info(db, user)
    await db.commit()
    return result


async def update_user(
        db: AsyncSession, actor: Principal, user_id: int, body: UserUpdate, password_hash: str | None):
    user = await db.scalar(select(User).where(User.id == user_id).with_for_update())
    if user is None:
        raise HTTPException(404, 'User not found')
    existing = await user_info(db, user)
    groups = await validate_groups(db, body.groups) if body.groups is not None else existing['groups']
    groups_changed = set(groups) != set(existing['groups'])
    removes_admin = ADMINS in existing['groups'] and ADMINS not in groups
    blocks_user = user.active and body.active is False
    if actor.id == user.id and (blocks_user or removes_admin):
        raise HTTPException(409, 'You cannot block yourself or remove your own admin access')
    if user.active and ADMINS in existing['groups'] and (blocks_user or removes_admin):
        others = await db.scalar(select(func.count()).select_from(User).join(Membership).where(
            User.active.is_(True), User.id != user.id, Membership.group_name == ADMINS))
        if not others:
            raise HTTPException(409, 'At least one active administrator is required')
    if body.active is not None:
        user.active = body.active
    if password_hash is not None:
        user.password_hash = password_hash
    if groups_changed:
        await db.execute(delete(Membership).where(Membership.user_id == user.id))
        db.add_all([Membership(user_id=user.id, group_name=name) for name in groups])
    if blocks_user or password_hash is not None or groups_changed:
        await revoke_sessions(db, user.id)
    result = await user_info(db, user)
    await db.commit()
    return result


async def reset_password(db: AsyncSession, username: str, password: str):
    password_hash = await password_work(hash_password, password)
    user = await db.scalar(select(User).where(User.username == username).with_for_update())
    if user is None:
        raise ValueError('User not found')
    user.password_hash = password_hash
    await revoke_sessions(db, user.id)
    await db.commit()


async def revoke_sessions(db: AsyncSession, user_id: int):
    await db.execute(delete(Session).where(Session.user_id == user_id))
