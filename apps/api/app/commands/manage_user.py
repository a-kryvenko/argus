"""Create dashboard users or reset passwords without shell-history secrets."""
import argparse
import asyncio
from getpass import getpass
from fastapi import HTTPException
from sqlalchemy import select, delete
from app.commands._runner import run_command
from app.db.session import get_session_factory, dispose_engine
from app.db.models.dashboard import User, Session
from app.dashboard_auth import hash_password
from app.routers.dashboard import UserCreate, create_user


async def manage(args, password):
    try:
        async with get_session_factory()() as db:
            if args.action == 'create':
                result = await create_user(db, UserCreate(username=args.username, password=password, groups=args.group))
                print(f"Created {result['username']}; groups: {', '.join(result['groups']) or '(none)'}")
            else:
                user = await db.scalar(select(User).where(User.username == args.username.lower()).with_for_update())
                if user is None:
                    raise ValueError('User not found')
                user.password_hash = hash_password(password)
                await db.execute(delete(Session).where(Session.user_id == user.id))
                await db.commit()
                print('Password reset; all sessions revoked')
    finally:
        await dispose_engine()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['create', 'reset-password'])
    parser.add_argument('username')
    parser.add_argument('--group', action='append', default=[], help='Group name; repeat to assign multiple groups')
    args = parser.parse_args()
    password = getpass('Password (12–256 characters): ')
    if not 12 <= len(password) <= 256:
        parser.error('Password must have 12–256 characters')
    if password != getpass('Repeat password: '):
        parser.error('Passwords do not match')
    try:
        asyncio.run(manage(args, password))
    except (ValueError, HTTPException) as exc:
        parser.exit(1, f"{getattr(exc, 'detail', str(exc))}\n")


if __name__ == '__main__':
    run_command(main)
