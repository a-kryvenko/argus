"""Create dashboard users and reset passwords."""
import asyncio
from getpass import getpass


def create(args):
    from fastapi import HTTPException

    password = _read_password(args.command_parser)
    try:
        return asyncio.run(_create(args, password))
    except (ValueError, HTTPException) as exc:
        args.command_parser.exit(1, f"{getattr(exc, 'detail', str(exc))}\n")


def reset_password(args):
    password = _read_password(args.command_parser)
    try:
        return asyncio.run(_reset_password(args, password))
    except ValueError as exc:
        args.command_parser.exit(1, f'{exc}\n')


def _read_password(parser):
    password = getpass('Password (12–256 characters): ')
    if not 12 <= len(password) <= 256:
        parser.error('Password must have 12–256 characters')
    if password != getpass('Repeat password: '):
        parser.error('Passwords do not match')
    return password


async def _create(args, password):
    from api.db.session import get_session_factory, dispose_engine
    from api.schemas.dashboard import UserCreate
    from api.services.authentication import create_user

    try:
        async with get_session_factory()() as db:
            result = await create_user(db, UserCreate(username=args.username, password=password, groups=args.group))
            print(f"Created {result['username']}; groups: {', '.join(result['groups']) or '(none)'}")
    finally:
        await dispose_engine()


async def _reset_password(args, password):
    from sqlalchemy import select, delete
    from api.db.session import get_session_factory, dispose_engine
    from api.db.models.dashboard import User, Session
    from api.services.authentication import hash_password

    try:
        async with get_session_factory()() as db:
            user = await db.scalar(select(User).where(User.username == args.username.lower()).with_for_update())
            if user is None:
                raise ValueError('User not found')
            user.password_hash = hash_password(password)
            await db.execute(delete(Session).where(Session.user_id == user.id))
            await db.commit()
            print('Password reset; all sessions revoked')
    finally:
        await dispose_engine()
