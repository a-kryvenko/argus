import gzip
import hashlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from clio.providers import donki


@pytest.mark.parametrize('content', [b'null', b'{}', b'[1]', b'<html>error</html>', b'[{"speed":NaN}]'])
def test_invalid_responses_are_not_empty_successes(content):
    with pytest.raises(ValueError):
        donki.decode(content)


def test_empty_feed_and_response_size_limit():
    assert donki.decode(b'[]') == []
    with pytest.raises(ValueError, match='8 MiB'):
        donki.decode(b' '*(donki.MAX_BYTES+1))


def test_public_api_request_is_bounded_and_uses_current_endpoint(monkeypatch):
    response = Mock(status_code=200)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.iter_content.return_value = [b'[{"activityID":"cme1","startTime":"2026-10-13T00:00Z"}]']
    get = Mock(return_value=response)
    monkeypatch.setattr(donki.requests, 'get', get)
    now = datetime(2026, 10, 13, tzinfo=UTC)
    raw, rows = donki.fetch('CME', now-timedelta(days=30), now)
    assert rows[0]['activityID'] == 'cme1' and raw.startswith(b'[')
    assert get.call_args.args[0] == 'https://ccmc.gsfc.nasa.gov/DONKI-API/get/CME'
    assert get.call_args.kwargs['allow_redirects'] is False
    assert get.call_args.kwargs['params'] == {'startDate': '2026-09-13', 'endDate': '2026-10-13'}


@pytest.mark.parametrize('row', [{}, {'activityID': 'cme1'},
    {'activityID': 'cme1', 'startTime': '2026-10-13T00:00:00'},
    {'activityID': 'cme1', 'startTime': 'invalid'}])
def test_unrecognized_records_cannot_silently_become_a_quiet_feed(row):
    with pytest.raises(ValueError):
        donki.validate('CME', [row])


def test_collection_retains_exact_bytes_empty_success_and_failure_independently(monkeypatch, tmp_path):
    monkeypatch.setenv("ARGUS_DONKI_ARCHIVE", str(tmp_path))
    import asyncio
    from clio.commands import collect_events
    saved = []
    session = AsyncMock()
    session.add = lambda row: saved.append(row)
    session.__aenter__.return_value = session
    monkeypatch.setattr(collect_events, 'get_db_session', lambda: session)
    monkeypatch.setattr(collect_events, 'load_observation_config', lambda: SimpleNamespace(
        solar_wind_events=SimpleNamespace(lookback_days=30)))
    raw = b'[ {"activityID": "cme1"} ]'
    def fetch(endpoint, *_):
        if endpoint == 'IPS':
            raise donki.requests.Timeout('private network details')
        return (raw, donki.decode(raw)) if endpoint == 'CME' else (b'[]', [])
    monkeypatch.setattr(donki, 'fetch', fetch)
    before = datetime.now(UTC)
    with pytest.raises(RuntimeError, match='IPS'):
        asyncio.run(collect_events.run(SimpleNamespace()))
    assert len(saved) == 4 and session.commit.await_count == 4
    cme = saved[0]
    assert gzip.decompress((tmp_path / cme.raw_path).read_bytes()) == raw
    assert cme.sha256 == hashlib.sha256(raw).hexdigest() and cme.received_at >= before
    assert gzip.decompress((tmp_path / saved[1].raw_path).read_bytes()) == b'[]' and saved[1].error is None
    assert saved[-1].raw_path is None and saved[-1].error == 'Timeout'

