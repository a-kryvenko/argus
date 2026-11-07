import asyncio
import pytest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from sqlalchemy.dialects import postgresql
from clio.services import solar_wind_events as service
from common.schemas.solar_wind_events import EventInputs, SourcePoll

NOW = datetime(2026, 10, 13, 12, tzinfo=UTC)


def test_input_reads_current_samples_and_latest_files(monkeypatch):
    poll = SimpleNamespace(endpoint='CME', id=uuid4(), received_at=NOW, sha256='a'*64,
        request_start=NOW-timedelta(days=30), request_end=NOW+timedelta(days=1), payload=[])
    sample = SimpleNamespace(observed_at=NOW-timedelta(minutes=1), received_at=NOW,
        metric='v', value=500, quality='unverified')
    result = lambda rows: Mock(scalars=lambda: Mock(all=lambda: rows))
    session = AsyncMock()
    polls = AsyncMock(return_value=EventInputs(schema_version=2, read_at=NOW, events=[], enlil_arrivals=[],
        polls=[SourcePoll(endpoint=poll.endpoint, poll_id=poll.id,
        received_at=poll.received_at, sha256=poll.sha256, request_start=poll.request_start,
        request_end=poll.request_end, record_count=0)]))
    monkeypatch.setattr(service, 'stored_event_inputs', polls)
    session.execute.side_effect = [result([sample]), Mock(mappings=lambda: Mock(all=lambda: []))]
    inputs = asyncio.run(service.load_event_inputs(session, NOW))
    assert inputs.schema_version == 2 and inputs.enlil_arrivals == [] and inputs.events == []
    assert inputs.polls[0].records == [] and inputs.l1_samples[0].value == 500
    queries = [c.args[0].compile(dialect=postgresql.dialect()) for c in session.execute.call_args_list]
    polls.assert_awaited_once_with(session, NOW)
    assert 'received_at <=' not in str(queries[0])
    assert set(queries[0].params['source_product_1']) == {'swpc.rtsw_mag', 'swpc.rtsw_plasma'}
    assert queries[0].params['observed_at_1'] == NOW-timedelta(hours=6)


@pytest.mark.parametrize('unavailable', [False, True])
def test_event_http_read_rolls_back_snapshot_on_success_and_failure(monkeypatch, unavailable):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from clio.routers import solar_wind_events as routes
    from clio.routers.errors import ObservationUnavailable
    from common.schemas.solar_wind_events import EventInputs

    cutoff = datetime.now(UTC) - timedelta(seconds=1)
    session = AsyncMock()
    load = AsyncMock(return_value=EventInputs(schema_version=2, read_at=cutoff,
                                              events=[], enlil_arrivals=[]))
    if unavailable:
        load.side_effect = ObservationUnavailable('DONKI original or receipt is unavailable or corrupt')
    monkeypatch.setattr(service, 'load_event_inputs', load)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_request_db_session] = lambda: session
    with TestClient(app) as client:
        response = client.get('/internal/v1/observations/solar-wind-events/inputs')
    assert response.status_code == (503 if unavailable else 200)
    if not unavailable:
        assert response.json()['schema_version'] == 2
        assert response.json()['events'] == []
        assert 'enlil_profiles' not in response.json()
        assert 'enlil_profile_status' not in response.json()
    assert session.execute.await_count == 2
    assert 'REPEATABLE READ READ ONLY' in str(session.execute.call_args_list[0].args[0])
    assert 'statement_timeout' in str(session.execute.call_args_list[1].args[0])
    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()


def test_database_failure_rolls_back_and_returns_503():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from sqlalchemy.exc import SQLAlchemyError

    session = AsyncMock()
    session.execute.side_effect = SQLAlchemyError('private connection details')
    app = FastAPI()
    from clio.routers import solar_wind_events as routes
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_request_db_session] = lambda: session
    response = TestClient(app).get('/internal/v1/observations/solar-wind-events/inputs')
    assert response.status_code == 503
    assert 'private connection details' not in response.text
    session.rollback.assert_awaited_once()
