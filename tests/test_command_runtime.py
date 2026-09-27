"""CLI boundaries report a logged/re-raised failure only once."""
import subprocess
import sys


def test_logged_failure_is_reported_once_and_success_returns_value():
    code = '''
import logging
import sys
from types import SimpleNamespace
import sentry_sdk
from common import runtime
runtime.get_config = lambda: SimpleNamespace(debug=False)
events = []
initialize = sentry_sdk.init
def isolated_init(**kwargs):
    kwargs.update(dsn='https://key@example.invalid/1', transport=events.append)
    return initialize(**kwargs)
sentry_sdk.init = isolated_init
assert runtime.run_command(lambda: 42) == 42
def fail():
    try:
        raise ValueError('original failure')
    except ValueError:
        logging.exception('failure recorded in logs')
        raise
try:
    runtime.run_command(fail)
except ValueError as exc:
    assert str(exc) == 'original failure'
    sys.excepthook(type(exc), exc, exc.__traceback__)
else:
    raise AssertionError('exception was swallowed')
sentry_sdk.flush()
assert len(events) == 1, len(events)
assert events[0]['exception']['values'][-1]['value'] == 'original failure'
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
