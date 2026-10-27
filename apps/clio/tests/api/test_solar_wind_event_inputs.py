import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from sqlalchemy.dialects import postgresql
from clio.routers import solar_wind_events as routes
from common.schemas.solar_wind_events import SourcePoll

NOW = datetime(2026, 10, 13, 12, tzinfo=UTC)


def test_input_reads_gate_native_samples_and_forward_cutoff_to_file_readers(monkeypatch):
    poll = SimpleNamespace(endpoint='CME', id=uuid4(), received_at=NOW, sha256='a'*64,
        request_start=NOW-timedelta(days=30), request_end=NOW+timedelta(days=1), payload=[])
    sample = SimpleNamespace(observed_at=NOW-timedelta(minutes=1), received_at=NOW,
        interval_end=None, metric='v', value=500, quality='unverified', spacecraft='ACE')
    result = lambda rows: Mock(scalars=lambda: Mock(all=lambda: rows))
    session = AsyncMock()
    polls = AsyncMock(return_value=([SourcePoll(endpoint=poll.endpoint, poll_id=poll.id,
        received_at=poll.received_at, sha256=poll.sha256, request_start=poll.request_start,
        request_end=poll.request_end, records=[])], {'CME': dict(received_at=NOW.isoformat(), error=None)}))
    profiles = AsyncMock(return_value=(None, None))
    monkeypatch.setattr(routes, 'donki_inputs', polls)
    monkeypatch.setattr(routes, 'enlil_input', profiles)
    session.execute.side_effect = [result([sample]), Mock(mappings=lambda: Mock(all=lambda: []))]
    inputs = asyncio.run(routes.load_event_inputs(session, NOW, NOW))
    assert inputs.schema_version == 2 and inputs.enlil_arrivals == [] and inputs.events == []
    assert inputs.polls[0].records == [] and inputs.l1_samples[0].value == 500
    queries = [c.args[0].compile(dialect=postgresql.dialect()) for c in session.execute.call_args_list]
    polls.assert_awaited_once_with(session, NOW)
    profiles.assert_awaited_once_with(session, NOW)
    assert queries[0].params['received_at_1'] == NOW
    assert set(queries[0].params['source_product_1']) == {'swpc.rtsw_mag', 'swpc.rtsw_plasma'}
    assert queries[0].params['observed_at_1'] == NOW-timedelta(hours=6)
