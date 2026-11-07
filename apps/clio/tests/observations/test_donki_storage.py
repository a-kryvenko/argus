import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from sqlalchemy.orm import Session

from clio.ingestion.extras import donki
from clio.ingestion.utils.file_observations import find
from clio.routers.solar_wind_events import donki_inputs

NOW = datetime(2026, 11, 3, 12, tzinfo=UTC)


def test_each_complete_response_is_uploaded_and_latest_empty_response_replaces_old_data(storage):
    for minute, raw in [(0, b'[{"activityID":"a"}]'), (1, b'[]')]:
        donki.record_poll('CME', raw=raw, observed_at=NOW+timedelta(minutes=minute),
            metadata=dict(request_start=NOW-timedelta(days=30), request_end=NOW+timedelta(days=1)))
    rows = find('donki.CME')
    assert len(rows) == len(storage.objects) == 2
    assert storage.objects['argus-private', rows[0]['key']][0] == b'[]'
    with Session(storage.engine) as session:
        class Reader:
            async def scalars(self, statement):
                return session.scalars(statement)
        polls, _ = asyncio.run(donki_inputs(Reader()))
    assert polls[0].records == [] and polls[0].received_at == NOW+timedelta(minutes=1)


@pytest.mark.parametrize('raw', [b'broken json', b'{}', b'[NaN]'])
def test_invalid_download_is_rejected_before_upload(storage, raw):
    with pytest.raises(ValueError):
        donki.record_poll('CME', raw=raw, metadata={})
    assert not storage.objects and not find('donki.CME')
