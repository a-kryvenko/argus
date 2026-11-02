from datetime import UTC, datetime, timedelta
import gzip
import json
from unittest.mock import AsyncMock, Mock
import asyncio

import pandas as pd
import pytest

from clio.ingestion import goes

NOW = datetime(2026, 9, 29, 12, 30, tzinfo=UTC)


@pytest.mark.parametrize('freshness,expected', [(5, 'partial'), (30, 'complete')])
def test_live_goes_uses_configured_freshness(monkeypatch, tmp_path, freshness, expected):
    from clio.config import load_observation_config
    config = load_observation_config().model_copy(deep=True)
    config.observations['goes'].freshness_minutes = freshness
    monkeypatch.setenv('ARGUS_GOES_ARCHIVE', str(tmp_path))
    frame = pd.DataFrame([{'timestamp': pd.Timestamp(NOW-timedelta(minutes=15)), 'goes_euv_256': 2.5}])
    monkeypatch.setattr(goes, 'source_frames', Mock(return_value=[frame]))
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.execute.return_value = Mock(scalar_one_or_none=Mock(return_value=1))
    monkeypatch.setattr(goes, 'get_db_session', lambda: session)
    monkeypatch.setattr(goes, 'snapshots', AsyncMock(return_value=[]))
    result = asyncio.run(goes.collect(config, mode='live', now=NOW))
    assert result['status'] == expected
    assert result['files']['goes']['failed'] == 0


def test_live_archives_source_columns_without_calibration(monkeypatch, tmp_path):
    monkeypatch.setenv('ARGUS_GOES_ARCHIVE', str(tmp_path))
    frame = pd.DataFrame([{'timestamp': pd.Timestamp(NOW-timedelta(minutes=10)),
                           'goes_euv_256': 2.5}])
    fetch = Mock(return_value=[frame])
    monkeypatch.setattr(goes, 'source_frames', fetch)
    records = goes.download_snapshots(NOW-timedelta(days=1), NOW, False)
    assert len(records) == 1
    row = records[0]
    samples = json.loads(gzip.decompress((tmp_path / row['raw_path']).read_bytes()))
    assert samples[0]['goes_euv_256'] == 2.5
    assert samples[0]['goes_euvs_quality_valid'] is True  # Legacy Prophet wire compatibility.
    assert 'goes_euvs_quality_valid' not in frame  # Absent from the provider contract.
    assert not {'s10', 'm10', 'y10'} & samples[0].keys()
    monkeypatch.setattr(goes, 'source_frames', Mock(side_effect=AssertionError('Fetched retained snapshot')))
    assert goes.download_snapshots(NOW-timedelta(days=1), NOW, False,
                                   {(row['slot_at'], row['source_product']): row}) == records


def test_archive_preserves_daily_receipts_and_rejects_revised_recovery(monkeypatch, tmp_path):
    monkeypatch.setenv('ARGUS_GOES_ARCHIVE', str(tmp_path))
    end = NOW.replace(hour=0, minute=0)
    start = end-timedelta(days=2)
    frame = pd.DataFrame({'timestamp': pd.date_range(start, periods=48, freq='h'), 'goes_euv_256': 2.5})
    monkeypatch.setattr(goes, 'source_frames', lambda *_, **kwargs: [frame])
    records = goes.download_snapshots(start, end, True)
    assert len(records) == 2
    expected = {(r['slot_at'], r['source_product']): r for r in records}
    row = records[0]
    path = tmp_path / row['raw_path']
    path.unlink()
    path.with_suffix(path.suffix + '.receipt.json').unlink()
    frame['goes_euv_256'] = 3.5
    with pytest.raises(ValueError, match='SHA256'):
        goes.download_snapshots(start, end, True, expected)
    assert not path.exists()


def test_backfill_recovers_local_receipts_without_redownloading_or_database_rows(monkeypatch, tmp_path):
    monkeypatch.setenv('ARGUS_GOES_ARCHIVE', str(tmp_path))
    end = NOW.replace(hour=0, minute=0)
    start = end-timedelta(days=2)
    frame = pd.DataFrame({'timestamp': pd.date_range(start, periods=48, freq='h'), 'goes_euv_256': 2.5})
    fetch = Mock(return_value=[frame])
    monkeypatch.setattr(goes, 'source_frames', fetch)
    before = goes.download_snapshots(start, end, True)
    fetch.reset_mock()
    after = goes.download_snapshots(start, end, True)
    fetch.assert_not_called()
    assert before == after


def test_only_missing_goes_days_are_passed_to_science_archive(monkeypatch, tmp_path):
    monkeypatch.setenv('ARGUS_GOES_ARCHIVE', str(tmp_path))
    end = NOW.replace(hour=0, minute=0)
    start = end-timedelta(days=3)
    frame = pd.DataFrame({'timestamp': pd.date_range(start, periods=72, freq='h'), 'goes_euv_256': 2.5})
    fetch = Mock(return_value=[frame])
    monkeypatch.setattr(goes, 'source_frames', fetch)
    records = goes.download_snapshots(start, end, True)
    missing = records[1]
    (tmp_path / missing['raw_path']).unlink()
    fetch.reset_mock()
    expected = {(r['slot_at'], r['source_product']): r for r in records}
    restored = goes.download_snapshots(start, end, True, expected)
    assert len(restored) == 3
    assert list(fetch.call_args.kwargs['days']) == [pd.Timestamp(missing['slot_at'])]
