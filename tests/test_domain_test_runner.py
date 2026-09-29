"""Lifecycle guarantees of the shared local/CI integration runner."""
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    'domain_runner', Path(__file__).resolve().parents[1] / 'scripts/testing/domain_storage.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_explicit_test_server_does_not_start_or_stop_docker(monkeypatch):
    monkeypatch.setenv('TEST_DATABASE_ADMIN_DSN', 'postgresql://disposable/test')
    monkeypatch.setattr(runner, 'run', lambda *a, **kw: pytest.fail('Unexpected Docker invocation'))
    with runner.postgres() as dsn:
        assert dsn == 'postgresql://disposable/test'


def test_temporary_server_is_removed_when_checks_fail(monkeypatch):
    monkeypatch.delenv('TEST_DATABASE_ADMIN_DSN', raising=False)
    commands = []
    def execute(*args, **kwargs):
        commands.append(args)
        return SimpleNamespace(stdout='127.0.0.1:15432\n')
    monkeypatch.setattr(runner, 'run', execute)
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0))
    with pytest.raises(RuntimeError, match='test failed'):
        with runner.postgres() as dsn:
            assert '@127.0.0.1:15432/postgres' in dsn
            raise RuntimeError('test failed')
    assert commands[-1][:2] == ('docker', 'stop')
    assert commands[-1][2] == commands[0][commands[0].index('--name') + 1]


def test_preflight_failure_stops_before_test_suite(monkeypatch):
    monkeypatch.setenv('TEST_DATABASE_ADMIN_DSN', 'postgresql://disposable/test')
    monkeypatch.setenv('PYTHONPATH', '/unrelated/environment')
    monkeypatch.setenv('DEBUG', 'false')
    monkeypatch.setenv('SENTRY_COLLECT_POINT', 'must-not-be-used')
    commands = []
    def fail(*args, **kwargs):
        commands.append(args)
        assert kwargs['env']['PYTHONPATH'] == ''
        assert kwargs['env']['DEBUG'] == 'true'
        assert kwargs['env']['SENTRY_COLLECT_POINT'] == ''
        raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(runner, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        runner.main()
    assert len(commands) == 1
    assert commands[0][1] == '-c'
