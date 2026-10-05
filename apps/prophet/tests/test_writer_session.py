from contextlib import nullcontext
from unittest.mock import Mock

import pytest
from argus_prophet.db import session


def test_all_writer_transactions_use_explicit_lock_connection_without_reconnecting(monkeypatch):
    conn = Mock(closed=False, broken=False)
    conn.transaction.side_effect = lambda: nullcontext()
    monkeypatch.setattr(session, 'open_connection', lambda **_: pytest.fail('Writer must not reconnect'))
    with session.transaction(conn) as first:
        assert first is conn
    conn.closed = True
    with pytest.raises(RuntimeError, match='lost'):
        with session.transaction(conn):
            pytest.fail('Lost writer must not continue')
    with pytest.raises(RuntimeError, match='require'):
        with session.transaction(None):
            pytest.fail('Unlocked writer must not continue')


def test_server_side_disconnect_propagates_without_reconnection(monkeypatch):
    conn = Mock(closed=False, broken=False)
    conn.transaction.side_effect = OSError('connection lost')
    monkeypatch.setattr(session, 'open_connection', lambda **_: pytest.fail('Writer must not reconnect'))
    with pytest.raises(OSError, match='connection lost'):
        with session.transaction(conn):
            pass
