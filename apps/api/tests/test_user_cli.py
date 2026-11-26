from unittest.mock import AsyncMock, Mock

import pytest

from api import cli
from api.commands import manage_user


@pytest.mark.parametrize('action,handler,options', [
    ('create', 'create', ['--group', 'admins', '--group', 'readers']),
    ('reset-password', 'reset_password', []),
])
def test_user_command_selects_handler(monkeypatch, action, handler, options):
    create = AsyncMock()
    reset_password = AsyncMock()
    monkeypatch.setattr(manage_user, '_create', create)
    monkeypatch.setattr(manage_user, '_reset_password', reset_password)
    monkeypatch.setattr(manage_user, 'getpass', lambda _: 'test-password-123')
    run_command = Mock(side_effect=lambda command, *, include_request_data: command())
    monkeypatch.setattr('common.runtime.run_command', run_command)

    cli.main(['user', action, 'alice', *options])

    selected = create if handler == 'create' else reset_password
    other = reset_password if handler == 'create' else create
    selected.assert_awaited_once()
    other.assert_not_called()
    args, password = selected.call_args.args
    assert args.username == 'alice'
    assert password == 'test-password-123'
    assert run_command.call_args.kwargs == {'include_request_data': False}
    if action == 'create':
        assert args.group == ['admins', 'readers']


@pytest.mark.parametrize('arguments', [
    ['user'],
    ['user', 'create'],
    ['user', 'reset-password', 'alice', '--group', 'admins'],
])
def test_invalid_user_arguments_fail_before_password_prompt(monkeypatch, arguments):
    def unexpected_prompt(_):
        pytest.fail('Invalid arguments must not prompt for a password')

    monkeypatch.setattr(manage_user, 'getpass', unexpected_prompt)
    with pytest.raises(SystemExit) as exc:
        cli.main(arguments)
    assert exc.value.code == 2


@pytest.mark.parametrize('passwords', [('short',), ('test-password-123', 'different-password')])
def test_invalid_password_does_not_call_handler(monkeypatch, passwords):
    handler = AsyncMock()
    answers = iter(passwords)
    monkeypatch.setattr(manage_user, '_create', handler)
    monkeypatch.setattr(manage_user, 'getpass', lambda _: next(answers))
    with pytest.raises(SystemExit) as exc:
        cli.main(['user', 'create', 'alice'])
    assert exc.value.code == 2
    handler.assert_not_called()
