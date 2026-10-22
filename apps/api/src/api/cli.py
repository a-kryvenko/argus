"""API server and maintenance commands."""
import argparse

from api.commands.manage_user import create, reset_password


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    serve = commands.add_parser('serve')
    serve.add_argument('--host', default='0.0.0.0')
    serve.add_argument('--port', type=int, default=8000)
    
    commands.add_parser('migrate', add_help=False)
    commands.add_parser('worker')

    user = commands.add_parser('user')
    user_commands = user.add_subparsers(dest='action', required=True)
    
    create_user = user_commands.add_parser('create')
    create_user.add_argument('username')
    create_user.add_argument(
        '--group',
        action='append',
        default=[],
        help='Group name; repeat to assign multiple groups'
    )
    create_user.set_defaults(command_parser=create_user)

    reset_user_password = user_commands.add_parser('reset-password')
    reset_user_password.add_argument('username')
    reset_user_password.set_defaults(command_parser=reset_user_password)

    args, remaining = parser.parse_known_args(argv)

    if args.command == 'migrate':
        from api.commands.migrate import run
        return run(remaining)

    if remaining:
        parser.error('Unrecognized arguments: ' + ' '.join(remaining))

    if args.command == 'worker':
        from common.runtime import run_command
        from api.worker import work
        return run_command(work)

    if args.command == 'user':
        from common.runtime import run_command
        if args.action == 'create':
            return run_command(lambda: create(args))
        if args.action == 'reset-password':
            return run_command(lambda: reset_password(args))

    if args.command == 'serve':
        import uvicorn
        return uvicorn.run(
            'api.main:app',
            host=args.host,
            port=args.port,
        )
