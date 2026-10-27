import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import postgresql

from api.services.monitoring import statistics as api_statistics


@pytest.mark.parametrize('size', [0, 1, 601])
def test_flush_batches_preserve_each_counter_and_duration(monkeypatch, size):
    hour = datetime(2026, 10, 22, tzinfo=UTC)
    monkeypatch.setattr(api_statistics, '_buffer', {
        (hour, f'/route{i}', 'GET', 200, 25): (i+1, i+.5) for i in range(size)})
    db = AsyncMock()
    db.__aenter__.return_value = db
    monkeypatch.setattr(api_statistics, 'get_db_session', lambda: db)
    asyncio.run(api_statistics.flush())
    inserts = [call.args[0].compile(dialect=postgresql.dialect())
               for call in db.execute.await_args_list if call.args[0].is_insert]
    assert [sum(key.startswith('count_m') for key in stmt.params) for stmt in inserts] == [
        min(500, size-offset) for offset in range(0, size, 500)]
    assert sum(value for stmt in inserts for key, value in stmt.params.items()
               if key.startswith('count_m')) == sum(range(1, size+1))
    assert sum(value for stmt in inserts for key, value in stmt.params.items()
               if key.startswith('duration_ms_m')) == sum(i+.5 for i in range(size))
    assert all('excluded.count' in str(stmt) and 'excluded.duration_ms' in str(stmt) for stmt in inserts)
    assert db.execute.await_count == len(inserts)+1  # Includes retention cleanup.
    assert api_statistics._buffer == {}
    db.commit.assert_awaited_once()


@pytest.mark.parametrize('route', ['/health/live', '/health/ready'])
def test_health_checks_do_not_enter_usage_statistics(monkeypatch, route):
    monkeypatch.setattr(api_statistics, '_buffer', {})
    api_statistics.record(route, 'GET', 200, 12)
    assert api_statistics._buffer == {}


@pytest.mark.parametrize('status', [200, 500])
def test_own_ping_probes_do_not_enter_usage_statistics(monkeypatch, status):
    monkeypatch.setattr(api_statistics, '_buffer', {})
    api_statistics.record('/ping', 'GET', status, 12, user_agent='Argus-Monitor/1')
    assert api_statistics._buffer == {}


@pytest.mark.parametrize('route,method,agent', [
    ('/ping', 'GET', ''),
    ('/ping', 'GET', 'Mozilla/5.0'),
    ('/ping', 'POST', 'Argus-Monitor/1'),
    ('/observations', 'GET', 'Argus-Monitor/1'),
    ('__unmatched__', 'GET', 'Argus-Monitor/1'),
])
def test_real_requests_still_enter_usage_statistics(monkeypatch, route, method, agent):
    monkeypatch.setattr(api_statistics, '_buffer', {})
    api_statistics.record(route, method, 200, 12, user_agent=agent)
    assert sum(count for count, _ in api_statistics._buffer.values()) == 1
