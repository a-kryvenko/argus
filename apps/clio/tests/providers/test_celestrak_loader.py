import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
import requests
from sqlalchemy.dialects import postgresql

from clio.config import load_observation_config
from clio.ingestion.adapters import CelesTrakHistoryAdapter
from clio.ingestion.live import load_celestrak_f107
from clio.ingestion.numeric_backfill import fetch_missing
from clio.ingestion.numeric_live import fetch_live
from clio.observations.measurements import upsert_measurements
from clio.providers import celestrak_loader as loader


HEADER = 'DATE,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81\n'
CSV = HEADER + (
    '2026-10-29,90,190,INT,290,390\n'
    '2026-10-30,91,191,OBS,291,111\n'
    '2026-10-31,92,192,OBS,292,112\n'
    '2026-11-01,93,193,PRD,293,113\n'
    '2026-12-01,94,194,PRM,294,114\n'
)
START = datetime(2026, 10, 30, tzinfo=UTC)
NOW = datetime(2026, 11, 1, tzinfo=UTC)


def test_observed_flux_and_published_trailing_mean_without_forecasts_or_interpolation():
    frame = loader.parse_celestrak_f107(CSV)
    assert frame.f10_7.tolist() == [91., 92.]
    assert frame.f10_7_81mean.tolist() == [111., 112.]
    assert frame.timestamp.tolist() == [pd.Timestamp('2026-10-30T20:00Z'), pd.Timestamp('2026-10-31T20:00Z')]


def test_missing_or_invalid_values_are_filtered_independently():
    frame = loader.parse_celestrak_f107(HEADER + (
        '2026-10-28,,1,OBS,2,110\n'
        '2026-10-29,91,1,OBS,2,inf\n'
        '2026-10-30,-1,1,OBS,2,0\n'
        '2026-10-31,NaN,1,OBS,2,bad\n'))
    assert frame.f10_7.dropna().tolist() == [91.]
    assert frame.f10_7_81mean.dropna().tolist() == [110.]
    assert len(frame) == 2


def test_forecast_only_file_returns_empty_typed_frame():
    frame = loader.parse_celestrak_f107(HEADER + '2026-11-01,93,193,PRD,293,113\n')
    assert frame.empty
    assert str(frame.timestamp.dtype) == 'datetime64[ns, UTC]'


def test_historical_observation_times():
    frame = loader.parse_celestrak_f107(HEADER + (
        '1991-05-31,91,191,OBS,291,111\n1991-06-01,92,192,OBS,292,112\n'))
    assert frame.timestamp.dt.hour.tolist() == [17, 20]


@pytest.mark.parametrize('text, match', [
    ('DATE,F10.7_OBS\n2026-10-31,92\n', 'missing'),
    (HEADER + 'bad-date,92,1,OBS,2,112\n', 'time data'),
    (HEADER + '2026-10-31,92,1,OBS,2,112\n' * 2, 'Duplicate'),
])
def test_invalid_format_and_ambiguous_dates_fail(text, match):
    with pytest.raises(ValueError, match=match):
        loader.parse_celestrak_f107(text)


def test_http_failure_is_propagated(monkeypatch):
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError('503')
    monkeypatch.setattr(loader.requests, 'get', Mock(return_value=response))
    with pytest.raises(requests.HTTPError):
        loader.fetch_celestrak_f107()


def test_backfill_batches_both_metrics_and_downloads_history_once(monkeypatch):
    response = Mock(text=CSV)
    download = Mock(return_value=response)
    monkeypatch.setattr(loader.requests, 'get', download)
    config = load_observation_config()
    policies = {m: config.observations[m] for m in loader.METRICS}
    adapter = CelesTrakHistoryAdapter()
    pending = {m: set(pd.date_range(START, NOW, inclusive='left')) for m in loader.METRICS}
    frame, attempts = fetch_missing(pending, policies, {'celestrak.f107_archive': adapter})
    assert frame.value.tolist() == [91., 92., 112.]
    assert len(attempts) == 1
    assert attempts[0]['accepted_measurements'] == 3
    # Subsequent date chunks in this backfill reuse the same source snapshot.
    assert adapter.fetch(pd.Timestamp('2026-10-31T20:00Z'), pd.Timestamp(NOW)).value.tolist() == [92., 112.]
    assert adapter.fetch(pd.Timestamp('2026-10-31T21:00Z'), pd.Timestamp(NOW)).empty
    download.assert_called_once_with(loader.CELESTRAK_HISTORY_URL, timeout=120)
    session = AsyncMock()
    asyncio.run(upsert_measurements(session, frame, replace_existing=False))
    values = session.execute.call_args.args[0].compile(dialect=postgresql.dialect()).params
    assert values['metric_m2'] == 'f10_7_81mean'
    assert values['value_m2'] == 112.


def test_live_clips_future_observations_and_fetches_both_metrics_once(monkeypatch):
    response = Mock(text=CSV)
    download = Mock(return_value=response)
    monkeypatch.setattr(loader.requests, 'get', download)
    cutoff = datetime(2026, 10, 31, 19, tzinfo=UTC)
    frame = load_celestrak_f107(START, cutoff)
    assert frame.value.tolist() == [91., 111.]
    assert frame.source_product.unique().tolist() == ['celestrak.f107']
    assert frame.received_at.notna().all()
    download.assert_called_once_with(loader.CELESTRAK_RECENT_URL, timeout=120)
    config = load_observation_config()
    fetch = AsyncMock(return_value=frame)
    fallback = AsyncMock(return_value=frame.iloc[:0])
    result, report = asyncio.run(fetch_live(list(loader.METRICS), config.observations,
        {'celestrak.f107': fetch, 'swpc.f107': fallback}, cutoff))
    fetch.assert_awaited_once()
    assert report['failed_metrics'] == []
    assert result.value.tolist() == [91., 111.]
