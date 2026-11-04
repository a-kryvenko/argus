import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from clio.config import load_observation_config
from clio.ingestion import collect_numeric as live_numeric
from clio.ingestion.utils.adapters import SOLFSMYHistoryAdapter
from clio.ingestion.utils.adapters import wide_to_measurements
from clio.ingestion.collect_numeric import load_solfsmy_means
from clio.providers import solfsmy as loader
from contextlib import asynccontextmanager

TEXT = '# YYYY DDD JD F10 F81c S10 S81c M10 M81c Y10 Y81c Ssrc\n' + (
    '2026 258 2461299.0 106.5 111.4 113.8 111.8 144.1 134.9 133.2 133.6 1E11\n'
    '2026 259 2461300.0 99.7 111.2 113.3 111.8 142.0 134.7 130.5 133.5 1E11\n')
NOW = datetime(2026, 11, 1, tzinfo=UTC)


def test_published_means_use_actual_window_end_without_recalculation():
    frame = loader.parse_solfsmy_means(TEXT)
    last = frame.loc[frame.timestamp.eq(pd.Timestamp('2026-10-26T12:00Z'))]
    assert last.drop(columns="timestamp").iloc[0].to_dict() == {
        'f10_7_81mean': 111.2, 's10_81mean': 111.8, 'm10_81mean': 134.7, 'y10_81mean': 133.5}
    assert len(frame) == 2


@pytest.mark.parametrize('text', ['2026 259 1 2', TEXT + TEXT.splitlines()[-1],
                                   TEXT.replace('2026 259', '2025 366')])
def test_invalid_or_duplicate_dates_and_format_fail(text):
    with pytest.raises(ValueError):
        loader.parse_solfsmy_means(text)


def test_invalid_mean_is_not_filled_from_daily_value():
    frame = loader.parse_solfsmy_means(TEXT.replace('134.7', '-1').replace('133.5', 'NaN'))
    last = frame.loc[frame.timestamp.eq(pd.Timestamp('2026-10-26T12:00Z'))]
    assert set(last.drop(columns='timestamp').iloc[0].dropna().index) == {'f10_7_81mean', 's10_81mean'}


def test_archive_download_once_and_live_receipt_keeps_window_date(monkeypatch):
    response = Mock(text=TEXT)
    download = Mock(return_value=response)
    monkeypatch.setattr(loader.requests, 'get', download)
    adapter = SOLFSMYHistoryAdapter()
    frame = adapter.fetch(NOW-timedelta(days=14), NOW)
    assert len(frame) == 8
    assert adapter.fetch(NOW-timedelta(days=5), NOW).empty
    download.assert_called_once_with(loader.SOLFSMY_URL, timeout=120)
    live = load_solfsmy_means(NOW-timedelta(days=14), NOW)
    assert live.observed_at.max() == pd.Timestamp('2026-10-26T12:00Z')
    assert live.source_product.eq('solfsmy.means').all()
    assert live.received_at.min() >= NOW


def test_collection_reads_delayed_mean_and_batches_solar_metrics(monkeypatch):
    frame = wide_to_measurements(loader.parse_solfsmy_means(TEXT)).assign(received_at=NOW)
    source = AsyncMock(return_value=frame)
    monkeypatch.setattr(live_numeric, 'collect_product', source)
    metrics = ['s10_81mean', 'm10_81mean', 'y10_81mean']
    report = asyncio.run(live_numeric.collect_numeric(load_observation_config(), metrics, now=NOW))
    assert report['failed_metrics'] == []
    assert report['downloaded_measurements'] == 6
    source.assert_awaited_once()
    assert source.call_args.args[-2] == NOW-timedelta(days=14)


def test_cleanup_retains_published_means_for_their_collection_horizon(storage, monkeypatch):
    from clio.ingestion import cleanup
    session = AsyncMock()
    session.execute.return_value = Mock(rowcount=0)
    @asynccontextmanager
    async def connection():
        yield session
    monkeypatch.setattr(cleanup, 'get_db_session', connection)
    config = load_observation_config().model_copy(deep=True)
    config.solar_wind_events.enabled = False
    asyncio.run(cleanup.cleanup(config, NOW))
    statement = session.execute.call_args_list[0].args[0].compile()
    assert NOW-timedelta(days=14) in statement.params.values()
    assert NOW-timedelta(days=6) in statement.params.values()
    assert all(metric in statement.params.values() for metric in ('s10_81mean', 'm10_81mean', 'y10_81mean'))
