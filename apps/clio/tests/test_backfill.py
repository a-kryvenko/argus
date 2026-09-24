import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from sqlalchemy.dialects import postgresql

from argus_clio.services import backfill as service, sensor_observations
from argus_clio.commands.backfill_observations import boundary

START = datetime(2026, 8, 31, tzinfo=UTC)
END = START + timedelta(days=1)


def test_history_clips_range_prefers_omni_and_excludes_ace_magnetic(monkeypatch):
    omni = pd.DataFrame({'issue_time': [START-timedelta(hours=1), START, END], 'v': [1., 420., 2.]})
    ace = pd.DataFrame({'issue_time': [START], 'v': [500.], 'n': [5.], 't': [-1e31], 'by': [99.]})
    monkeypatch.setattr(service.OMNIWeb_Loader, 'load', Mock(return_value=omni))
    monkeypatch.setattr(service.SPDF_Loader, 'load', Mock(return_value=ace))
    monkeypatch.setattr(service, 'get_config', lambda: object())
    monkeypatch.setattr(service, '_download_history', Mock(side_effect=OSError('archive unavailable')))
    frame, errors = service.load_history(START, END)
    assert frame.set_index('metric').value.to_dict() == {'v': 420., 'n': 5.}
    assert set(frame.observed_at) == {START}
    assert errors == ['solar indices: archive unavailable']


def test_all_sources_unavailable_fails(monkeypatch):
    monkeypatch.setattr(service.OMNIWeb_Loader, 'load', Mock(side_effect=RuntimeError('missing')))
    monkeypatch.setattr(service.SPDF_Loader, 'load', Mock(side_effect=ValueError('missing')))
    monkeypatch.setattr(service, 'get_config', lambda: object())
    monkeypatch.setattr(service, '_download_history', Mock(side_effect=OSError('missing')))
    with pytest.raises(RuntimeError, match='No backfill sources'):
        service.load_history(START, END)


def test_insert_only_preserves_existing_values_and_receipts():
    session = AsyncMock()
    frame = pd.DataFrame([{'metric': 'v', 'value': 420., 'observed_at': START}])
    asyncio.run(sensor_observations._upsert_measurements(session, frame, track_receipt=False, replace_existing=False))
    assert session.execute.await_count == 1
    sql = str(session.execute.call_args.args[0].compile(dialect=postgresql.dialect()))
    assert 'ON CONFLICT ON CONSTRAINT uq_measurement_metric DO NOTHING' in sql


def test_backfill_rebuilds_from_stored_values_and_reports_raw_gaps(monkeypatch):
    frame = pd.DataFrame([{'metric': 'v', 'value': 420., 'observed_at': START}])
    monkeypatch.setattr(service, 'load_history', lambda *_: (frame, []))
    insert = AsyncMock()
    monkeypatch.setattr(service, '_upsert_measurements', insert)
    session = AsyncMock()
    stored = [('v', 450., START)]
    session.execute.return_value = Mock(all=lambda: stored)
    normalize = Mock(return_value=pd.DataFrame())
    monkeypatch.setattr(service, 'normalize_measurements', normalize)
    monkeypatch.setattr(service, '_upsert_normalized_observations', AsyncMock())
    result = asyncio.run(service.backfill(session, START, END))
    assert normalize.call_args.args[0].value.tolist() == [450.]
    assert insert.call_args.kwargs == {'track_receipt': False, 'replace_existing': False}
    assert result['missing_observed_hours']['v'] == 23
    assert result['normalized_hours'] == 0
    session.commit.assert_awaited_once()


def test_boundary_accepts_dates_and_rejects_naive_hours():
    import argparse
    assert boundary('2026-08-31') == START
    assert boundary('2026-08-31T02:00:00+02:00') == START
    with pytest.raises(argparse.ArgumentTypeError):
        boundary('2026-08-31T01:00:00')
