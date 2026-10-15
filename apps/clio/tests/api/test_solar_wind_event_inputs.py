import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from sqlalchemy.dialects import postgresql
from clio.routers.solar_wind_events import load_event_inputs

NOW = datetime(2026, 10, 13, 12, tzinfo=UTC)


def test_input_reads_gate_receipts_keep_native_samples_and_latest_complete_feeds():
    poll = SimpleNamespace(endpoint='CME', id=uuid4(), received_at=NOW, sha256='a'*64,
        request_start=NOW-timedelta(days=30), request_end=NOW+timedelta(days=1), payload=[])
    sample = SimpleNamespace(observed_at=NOW-timedelta(minutes=1), received_at=NOW,
        interval_end=None, metric='v', value=500, quality='unverified', spacecraft='ACE')
    result = lambda rows: Mock(scalars=lambda: Mock(all=lambda: rows))
    session = AsyncMock()
    session.execute.side_effect = [result([poll]), Mock(all=lambda: [SimpleNamespace(
        endpoint='CME', received_at=NOW, error=None)]), result([sample]), Mock(mappings=lambda: Mock(all=lambda: [])), Mock(scalars=lambda: Mock(first=lambda: None)), Mock(first=lambda: None)]
    inputs = asyncio.run(load_event_inputs(session, NOW, NOW))
    assert inputs.schema_version == 2 and inputs.enlil_arrivals == [] and inputs.events == []
    assert inputs.polls[0].records == [] and inputs.l1_samples[0].value == 500
    queries = [c.args[0].compile(dialect=postgresql.dialect()) for c in session.execute.call_args_list]
    assert 'DISTINCT ON (clio.donki_poll.endpoint)' in str(queries[0])
    assert queries[0].params['received_at_1'] == NOW
    assert queries[2].params['received_at_1'] == NOW
    assert set(queries[2].params['source_product_1']) == {'swpc.rtsw_mag', 'swpc.rtsw_plasma'}
    assert queries[2].params['observed_at_1'] == NOW-timedelta(hours=6)
