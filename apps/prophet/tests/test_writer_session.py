from contextlib import nullcontext
from unittest.mock import Mock

import pytest
from argus_prophet.db import session
from argus_prophet.exports import write_csv


def test_all_writer_transactions_use_lock_session_without_reconnecting(monkeypatch):
    conn = Mock(closed=False, broken=False)
    conn.transaction.side_effect = lambda: nullcontext()
    monkeypatch.setattr(session, 'open_connection', lambda **_: pytest.fail('Writer must not reconnect'))
    with session.writer_session(conn):
        with session.connect(writing=True) as first:
            assert first is conn
        with session.connect() as reader:
            assert reader is conn
        conn.closed = True
        with pytest.raises(RuntimeError, match='lost'):
            with session.connect(writing=True):
                pytest.fail('Lost writer must not continue')
    with pytest.raises(RuntimeError, match='require'):
        with session.connect(writing=True):
            pytest.fail('Unlocked writer must not continue')


def test_server_side_disconnect_propagates_without_reconnection(monkeypatch):
    conn = Mock(closed=False, broken=False)
    conn.transaction.side_effect = OSError('connection lost')
    monkeypatch.setattr(session, 'open_connection', lambda **_: pytest.fail('Writer must not reconnect'))
    with session.writer_session(conn):
        with pytest.raises(OSError, match='connection lost'):
            with session.connect(writing=True):
                pass


def test_lost_lock_before_csv_replace_keeps_previous_file(tmp_path):
    path = tmp_path / 'live.csv'
    path.write_bytes(b'previous')
    def lost():
        raise RuntimeError('lock lost')
    with pytest.raises(RuntimeError, match='lock lost'):
        write_csv(path, b'new', 'release', before_replace=lost)
    assert path.read_bytes() == b'previous'
    assert not list(tmp_path.glob('*.tmp'))
