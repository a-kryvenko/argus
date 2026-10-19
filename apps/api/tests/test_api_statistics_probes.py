import pytest

from api.services.monitoring import statistics as api_statistics


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
