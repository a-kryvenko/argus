import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from app.services import project_monitoring as monitoring
from app.services.edge_traffic import read_batch, aggregate, traffic_summary
from app.routers.dashboard import project_monitoring, project_traffic

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
    buckets, invalid = aggregate([event(), event('site', 404), event('monitor'), b'{broken',
        event(seconds='NaN'), event(stamp=NOW-timedelta(days=31)), event(stamp=NOW+timedelta(days=1))], NOW)
    assert invalid == 2
    assert sum(v[0] for v in buckets.values()) == 4  # two requests, minute + hour
    assert {key[2] for key in buckets} == {'api', 'site'}
    assert {key[4] for key in buckets} == {50}


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


def test_host_metrics_are_unknown_without_host_mounts(monkeypatch):
    monkeypatch.delenv('MONITORING_HOST_PROC', raising=False)
    assert monitoring.host_metrics()['status'] == 'unknown'


def test_host_metrics_use_delta_cpu_and_available_memory(tmp_path, monkeypatch):
    (tmp_path/'stat').write_text('cpu  100 0 100 800 0 0 0 0 0 0\n')
    (tmp_path/'meminfo').write_text('MemTotal: 1000 kB\nMemAvailable: 250 kB\n')
    monkeypatch.setenv('MONITORING_HOST_PROC', str(tmp_path))
    monkeypatch.setenv('MONITORING_DISK_PATH', str(tmp_path))
    result = monitoring.host_metrics({'cpu_sample': [900, 750]})
    assert result['cpu_percent'] == 50 and result['memory_percent'] == 75
    assert result['disk_free_bytes'] > 0


def test_individual_probe_failure_does_not_hide_other_services(monkeypatch):
    monkeypatch.setenv('OBSERVATIONS_URL', 'http://clio')
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'test')
    async def run():
        def respond(request):
            if request.url.host == 'clio':
                return httpx.Response(503)
            return httpx.Response(200)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            clio, site, observations = await asyncio.gather(
                monitoring.probe_http(client, 'Clio', 'http://clio', '/health/ready'),
                monitoring.probe_http(client, 'Website', 'http://site', '/'),
                monitoring.observation_status(client))
        assert clio['status'] == 'down'
        assert site['status'] == 'ok'
        assert observations['status'] == 'unknown'
    asyncio.run(run())


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
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as client:
            return await monitoring.forecast_status(client, 'dst')
    result = asyncio.run(run())
    assert result['status'] == 'warning'
    assert result['current_release']['published_at'] == NOW.isoformat().replace('+00:00', 'Z')
    assert 'private' not in result['latest_attempt']['error']


def test_traffic_heartbeat_must_be_recent_even_if_reader_is_running():
    from unittest.mock import Mock
    db = AsyncMock()
    db.get.return_value = SimpleNamespace(checked_at=datetime.now(UTC), payload={
        'status': 'ok', 'last_event_at': (datetime.now(UTC)-timedelta(minutes=3)).isoformat()})
    db.scalars.return_value = Mock(all=Mock(return_value=[]))
    db.scalar.return_value = 0
    result = asyncio.run(project_traffic('hour', db))
    assert result.data['status'] == 'unknown' and result.data['stale']
