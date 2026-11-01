import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from api.services.monitoring import project as monitoring
from api.services.monitoring.edge_traffic import read_batch, aggregate, traffic_summary
from api.routers.dashboard import project_monitoring, project_traffic

NOW = datetime(2026, 9, 17, 12, tzinfo=UTC)


def event(channel='api', status=200, seconds='0.042', stamp=NOW):
    return json.dumps({'channel': channel, 'status': status, 'seconds': seconds, 'time': stamp.timestamp()}).encode()


def test_traffic_tailer_handles_partial_writes_restart_and_rotation(tmp_path):
    log = tmp_path / 'traffic.jsonl'
    first, second, third = event(), event('site'), event(status=502)
    log.write_bytes(first + b'\n' + second[:15])
    lines, cursor, caught_up = read_batch(log, {})
    assert lines == [first] and not caught_up
    with log.open('ab') as stream:
        stream.write(second[15:] + b'\n')
    # Rotate before the consumer catches up with the previous inode.
    log.rename(tmp_path / 'traffic.jsonl.1')
    log.write_bytes(third + b'\n')
    lines, cursor, caught_up = read_batch(log, cursor)
    assert lines == [second] and not caught_up
    lines, cursor, caught_up = read_batch(log, cursor)
    assert lines == [third] and caught_up
    assert read_batch(log, cursor)[0] == []


def test_traffic_filters_probes_invalid_records_and_retention():
    buckets, invalid, last_event = aggregate([event(), event('site', 404), event('monitor'), b'{broken',
        event(seconds='NaN'), event(stamp=NOW-timedelta(days=31)), event(stamp=NOW+timedelta(days=1))], NOW)
    assert invalid == 2
    assert sum(v[0] for v in buckets.values()) == 4  # two requests, minute + hour
    assert {key[2] for key in buckets} == {'api', 'site'}
    assert {key[4] for key in buckets} == {50}
    assert last_event == (NOW-timedelta(days=31)).isoformat()


@pytest.mark.parametrize('lines,invalid,last_event', [
    ([event('monitor')], 0, NOW.isoformat()),
    ([b'{"channel":"monitor"}'], 0, None),
    ([event(), b'{broken', event(stamp=NOW+timedelta(seconds=6))], 1, NOW.isoformat()),
    ([json.dumps({'time': NOW.timestamp()}).encode()], 1, NOW.isoformat()),
    ([], 0, None),
])
def test_traffic_heartbeat_includes_probes_and_ignores_future_timestamps(lines, invalid, last_event):
    _, actual_invalid, actual_last_event = aggregate(lines, NOW)
    assert (actual_invalid, actual_last_event) == (invalid, last_event)


def test_traffic_summary_separates_channels_and_fills_empty_intervals():
    rows = [SimpleNamespace(time=NOW, channel='site', count=2, duration_ms=20, status=502, bucket_ms=10)]
    result = traffic_summary(rows, NOW-timedelta(minutes=1), NOW, 'minute')
    assert result['api']['summary']['requests'] == 0
    assert result['site']['summary']['errors_5xx'] == 2
    assert [p['requests'] for p in result['site']['points']] == [0, 2]


def test_old_project_snapshot_is_never_healthy():
    db = AsyncMock()
    db.get.return_value = SimpleNamespace(checked_at=datetime.now(UTC)-timedelta(minutes=3), payload={'status': 'ok'})
    result = asyncio.run(project_monitoring(db))
    assert result.data['status'] == 'unknown' and result.data['stale']


def test_missing_project_snapshot_is_unknown():
    db = AsyncMock()
    db.get.return_value = None
    result = asyncio.run(project_monitoring(db))
    assert result.data['status'] == 'unknown'


def forecast_payload(product='dst', **changes):
    return {'product': product, 'assessed_at': NOW.isoformat(), 'current_release': None,
            'release_age_hours': None, 'existing_public_max_age_hours': None,
            'freshness': 'within_age_limit', 'latest_attempt': None,
            'latest_attempt_artifacts': [], **changes}


@pytest.mark.parametrize('forecast_unavailable', [False, True])
def test_product_collection_makes_two_authenticated_status_requests(monkeypatch, forecast_unavailable):
    monkeypatch.setenv('OBSERVATIONS_URL', 'http://clio')
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'observation-secret')
    monkeypatch.setenv('FORECASTS_URL', 'http://prophet')
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'forecast-secret')
    requests = []
    def respond(request):
        requests.append((str(request.url), request.headers['Authorization']))
        if request.url.host == 'clio':
            return httpx.Response(200, json={'success': True, 'data': {
                'status': 'ok', 'measurements': []}})
        return httpx.Response(503 if forecast_unavailable else 200,
            json={'products': [forecast_payload(p) for p in monitoring.PRODUCT_ARTIFACTS]})
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: client(
        transport=httpx.MockTransport(respond), **kwargs))
    result = asyncio.run(monitoring.collect())
    assert result['status'] == ('degraded' if forecast_unavailable else 'ok')
    assert result['observations']['status'] == 'ok'
    assert {item['status'] for item in result['forecasts']} == {'unknown' if forecast_unavailable else 'ok'}
    assert set(result) == {'status', 'observations', 'forecasts'}
    assert {item['product'] for item in result['forecasts']} == set(monitoring.PRODUCT_ARTIFACTS)
    assert sorted(requests) == [
        ('http://clio/internal/v1/observations/status', 'Bearer observation-secret'),
        ('http://prophet/internal/v1/forecasts/status', 'Bearer forecast-secret')]


def test_legacy_infrastructure_status_is_removed_from_dashboard():
    payload = {'status': 'degraded', 'host': {'status': 'warning'},
        'services': [{'name': 'Redis', 'status': 'down'}],
        'observations': {'status': 'ok'}, 'forecasts': [
            {'product': product, 'status': 'ok'} for product in monitoring.PRODUCT_ARTIFACTS]}
    db = AsyncMock()
    db.get.return_value = SimpleNamespace(checked_at=datetime.now(UTC), payload=payload)
    result = asyncio.run(project_monitoring(db)).data
    assert result['status'] == 'ok'
    assert 'host' not in result and 'services' not in result


def test_forecast_keeps_release_when_latest_generation_failed(monkeypatch):
    monkeypatch.setenv('FORECASTS_URL', 'http://prophet')
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'test')
    identifier = '00000000-0000-4000-8000-000000000001'
    payload = {'product': 'dst', 'assessed_at': NOW.isoformat(),
        'current_release': {'release_id': identifier, 'run_id': identifier, 'issue_time': NOW.isoformat(),
                            'published_at': NOW.isoformat(), 'started_at': NOW.isoformat()},
        'release_age_hours': 0, 'existing_public_max_age_hours': 24, 'freshness': 'within_age_limit',
        'latest_attempt': {'run_id': identifier, 'status': 'failed', 'started_at': NOW.isoformat(),
                           'error': 'private database credentials'}, 'latest_attempt_artifacts': []}
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={'products': [
            payload if p == 'dst' else forecast_payload(p) for p in monitoring.PRODUCT_ARTIFACTS]}))) as client:
            return await monitoring.forecast_status(client)
    result = next(item for item in asyncio.run(run()) if item['product'] == 'dst')
    assert result['status'] == 'warning'
    assert result['current_release']['published_at'] == NOW.isoformat().replace('+00:00', 'Z')
    assert 'private' not in result['latest_attempt']['error']


def test_traffic_heartbeat_must_be_recent_even_if_reader_is_running(monkeypatch):
    monkeypatch.setenv('MONITORING_TRAFFIC_LOG', '/var/log/nginx/traffic.jsonl')
    from unittest.mock import Mock
    db = AsyncMock()
    db.get.return_value = SimpleNamespace(checked_at=datetime.now(UTC), payload={
        'status': 'ok', 'last_event_at': (datetime.now(UTC)-timedelta(minutes=3)).isoformat()})
    db.scalars.return_value = Mock(all=Mock(return_value=[]))
    db.scalar.return_value = 0
    result = asyncio.run(project_traffic('hour', db))
    assert result.data['status'] == 'unknown' and result.data['stale']


@pytest.mark.parametrize('count', [0, 7])
def test_recent_error_count_serializes_as_json_number(count, monkeypatch):
    monkeypatch.setenv('MONITORING_TRAFFIC_LOG', '/var/log/nginx/traffic.jsonl')
    from decimal import Decimal
    from unittest.mock import Mock
    db = AsyncMock()
    now = datetime.now(UTC)
    db.get.return_value = SimpleNamespace(checked_at=now, payload={
        'status': 'ok', 'last_event_at': now.isoformat()})
    db.scalars.return_value = Mock(all=Mock(return_value=[]))
    # PostgreSQL SUM(bigint) returns Decimal even when its value is zero.
    db.scalar.return_value = Decimal(count)
    result = asyncio.run(project_traffic('hour', db))
    payload = json.loads(result.model_dump_json())
    assert type(payload['data']['recent_errors_5xx']) is int
    assert payload['data']['recent_errors_5xx'] == count


def test_unconfigured_traffic_is_disabled_without_reading_old_counts(monkeypatch):
    monkeypatch.delenv('MONITORING_TRAFFIC_LOG', raising=False)
    db = AsyncMock()
    result = asyncio.run(project_traffic('hour', db))
    assert result.data['status'] == 'disabled'
    assert result.data['stale'] is False
    assert result.data['channels'] is None
    assert result.data['recent_errors_5xx'] == 0
    db.get.assert_not_awaited()
