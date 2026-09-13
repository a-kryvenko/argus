import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError

from argus_clio.db.models import ObservationSourceStatus
from argus_clio.services import collection_status as service
from argus_clio.routers import collection_status as routes

NOW = datetime(2026, 9, 8, 12, tzinfo=UTC)


def status(**changes):
    fields = dict(source_id='solar_wind_mag', last_attempt_at=NOW-timedelta(seconds=10),
        last_completed_at=NOW-timedelta(seconds=5), last_response_at=NOW-timedelta(seconds=6),
        last_success_at=NOW-timedelta(seconds=5), last_error_at=None, last_error_code=None,
        last_error_message=None, consecutive_failures=0, latest_observation_at=NOW-timedelta(minutes=3),
        latest_interval_end=None, data_quality='complete')
    return ObservationSourceStatus(**(fields | changes))


@pytest.mark.parametrize('changes, expected', [
    ({}, 'ok'),
    ({'latest_observation_at': NOW-timedelta(hours=1)}, 'source_delayed'),
    ({'last_attempt_at': NOW-timedelta(minutes=4), 'last_completed_at': NOW-timedelta(minutes=3)}, 'collector_overdue'),
    ({'last_attempt_at': NOW-timedelta(seconds=121), 'last_completed_at': None}, 'collector_stalled'),
    ({'consecutive_failures': 2}, 'collection_error'),
    ({'data_quality': 'partial'}, 'data_partial'),
    ({'data_quality': 'unavailable'}, 'data_unavailable'),
    ({'last_completed_at': None, 'last_success_at': None}, 'collecting'),
])
def test_distinguishes_collection_failures_and_source_delays(changes, expected):
    result = service.describe('solar_wind_mag', status(**changes), NOW)
    assert result['status'] == expected


def test_old_attempt_takes_priority_over_old_error_and_data():
    record = status(last_attempt_at=NOW-timedelta(minutes=5), last_completed_at=NOW-timedelta(minutes=4),
                    consecutive_failures=3, latest_observation_at=NOW-timedelta(hours=2))
    result = service.describe('solar_wind_mag', record, NOW)
    assert result['status'] == 'collector_overdue'
    assert result['data_status'] == 'delayed' and result['consecutive_failures'] == 3


def test_polling_success_does_not_refresh_measurement_age():
    record = status(last_success_at=NOW, last_response_at=NOW, latest_observation_at=NOW-timedelta(hours=1))
    result = service.describe('solar_wind_mag', record, NOW)
    assert result['data_age_seconds'] == 3600 and result['status'] == 'source_delayed'


def test_index_freshness_uses_interval_end_and_its_own_schedule():
    record = status(source_id='dst', last_attempt_at=NOW-timedelta(minutes=4), last_completed_at=NOW-timedelta(minutes=3),
                    latest_observation_at=NOW-timedelta(hours=2), latest_interval_end=NOW-timedelta(hours=1))
    result = service.describe('dst', record, NOW)
    assert result['status'] == 'ok' and result['data_age_seconds'] == 3600
    record.latest_interval_end = NOW+timedelta(hours=1)
    assert service.describe('kp', record, NOW)['data_age_seconds'] == 0


def test_recovery_keeps_last_error_but_reports_ok():
    result = service.describe('solar_wind_mag', status(last_error_at=NOW-timedelta(minutes=5),
                               last_error_code='source_timeout', last_error_message='Request timed out.'), NOW)
    assert result['status'] == 'ok' and result['last_error_code'] == 'source_timeout'


def test_unknown_source_progress_is_not_a_stopped_process_claim():
    result = service.describe('kp', None, NOW)
    assert result['status'] == 'not_started'
    assert result['collector_status'] == 'unknown'
    assert result['last_attempt_at'] is None


@pytest.mark.parametrize('error, code', [
    (requests.Timeout('password=secret'), 'source_timeout'),
    (requests.ConnectionError('https://user:secret@host'), 'source_connection_error'),
    (ValueError('raw private response'), 'invalid_response'),
    (SQLAlchemyError('SELECT secret'), 'storage_error'),
    (RuntimeError('private/path'), 'collection_error'),
])
def test_public_errors_do_not_expose_exception_details(error, code):
    result = service.public_error(error)
    assert result[0] == code and str(error) not in result[1]


def test_http_status_is_safe_to_publish():
    response = requests.Response(); response.status_code = 503
    code, message = service.public_error(requests.HTTPError('secret', response=response))
    assert code == 'source_http_error' and '503' in message and 'secret' not in message


def test_failure_is_recorded_after_rollback_and_cannot_overwrite_newer_attempt():
    observations, marker = AsyncMock(), AsyncMock()
    context = AsyncMock(); context.__aenter__.return_value = marker
    async def fail():
        async with service.track_attempt('kp', observations, lambda: context):
            marker.commit.assert_awaited_once()  # Attempt visible before source work starts.
            raise requests.Timeout('offline')
    with pytest.raises(requests.Timeout):
        asyncio.run(fail())
    observations.rollback.assert_awaited_once()
    observations.commit.assert_not_awaited()
    assert marker.commit.await_count == 2
    compiled = marker.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    assert 'last_attempt_at =' in str(compiled)
    assert compiled.params['last_error_code'] == 'source_timeout'


def test_success_commits_status_and_measurements_together():
    observations, marker = AsyncMock(), AsyncMock()
    context = AsyncMock(); context.__aenter__.return_value = marker
    async def succeed():
        async with service.track_attempt('kp', observations, lambda: context) as attempt:
            attempt.received([{'interval_start': NOW-timedelta(hours=3), 'interval_end': NOW,
                               'value': 3.33, 'quality': 'unverified'}])
            observations.commit.assert_not_awaited()
    asyncio.run(succeed())
    observations.commit.assert_awaited_once()
    compiled = observations.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    assert compiled.params['consecutive_failures'] == 0
    assert compiled.params['latest_interval_end'] == NOW


def test_api_returns_all_sources_in_empty_storage_without_fetching(monkeypatch):
    app = FastAPI(); app.include_router(routes.router)
    session = AsyncMock(); session.execute.return_value = Mock(scalars=Mock(return_value=[]))
    app.dependency_overrides[routes.get_db_session] = lambda: session
    with TestClient(app) as client:
        response = client.get('/internal/v1/observations/status')
    assert response.status_code == 200 and response.headers['cache-control'] == 'no-store'
    data = response.json()['data']
    assert data['status'] == 'degraded' and len(data['sources']) == 4
    assert all(source['status'] == 'not_started' for source in data['sources'].values())
