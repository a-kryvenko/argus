from contextlib import nullcontext
from unittest.mock import Mock

import pytest
from prophet.db.locks import Writer, transaction

pytestmark = pytest.mark.anyio


async def test_writer_transactions_use_lock_connection_and_reject_lost_writer():
    connection = Mock(closed=False, invalidated=False)
    session = Mock()
    session.begin.side_effect = lambda: nullcontext()
    writer = Writer(session, connection)
    async with transaction(writer) as first:
        assert first is session
    connection.invalidated = True
    with pytest.raises(RuntimeError, match='lost'):
        async with transaction(writer):
            pytest.fail('Lost writer must not continue')
    with pytest.raises(RuntimeError, match='require'):
        async with transaction(None):
            pytest.fail('Unlocked writer must not continue')


async def test_server_side_disconnect_permanently_invalidates_writer():
    connection = Mock(closed=False, invalidated=False)
    session = Mock()
    session.begin.side_effect = lambda: nullcontext()
    writer = Writer(session, connection)
    with pytest.raises(OSError, match='connection lost'):
        async with transaction(writer):
            connection.invalidated = True
            raise OSError('connection lost')
    # Revalidating a connection must never revive the old writer.
    connection.invalidated = False
    with pytest.raises(RuntimeError, match='lost'):
        async with transaction(writer):
            pytest.fail('Lost writer reconnected')
