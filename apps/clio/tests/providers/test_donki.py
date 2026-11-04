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


def test_collection_retains_exact_bytes_and_empty_success_but_skips_failures(storage, monkeypatch, tmp_path):
    monkeypatch.setenv("ARGUS_DONKI_ARCHIVE", str(tmp_path))
    import asyncio
    from clio.ingestion.extras import donki as collector
    policy = SimpleNamespace(lookback_days=30)
    raw = b'[ {"activityID": "cme1"} ]'
    def fetch(endpoint, *_):
        if endpoint == 'IPS':
            raise donki.requests.Timeout('private network details')
        return (raw, donki.decode(raw)) if endpoint == 'CME' else (b'[]', [])
    monkeypatch.setattr(donki, 'fetch', fetch)
    before = datetime.now(UTC)
    with pytest.raises(RuntimeError, match='IPS'):
        asyncio.run(collector.collect(policy))
    from sqlalchemy import select
    from clio.db.models.file_observation import FileObservation
    with storage.engine.connect() as connection:
        saved = [SimpleNamespace(**row) for row in connection.execute(
            select(FileObservation.__table__).order_by(FileObservation.received_at)).mappings()]
    assert len(saved) == 3
    cme = saved[0]
    assert storage.objects['argus-private', storage.files(id=cme.file_id)[0]['key']][0] == raw
    assert cme.details['sha256'] == hashlib.sha256(raw).hexdigest() and cme.received_at >= before
    assert storage.objects['argus-private', storage.files(id=saved[1].file_id)[0]['key']][0] == b'[]'
    assert 'IPS' not in {row.source_product for row in saved}
    assert len(storage.objects) == 3

