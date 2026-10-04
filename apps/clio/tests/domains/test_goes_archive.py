from datetime import UTC, datetime, timedelta
import json
from unittest.mock import Mock

import pandas as pd
import pytest

from clio.domains import goes

NOW = datetime(2026, 9, 29, 12, 30, tzinfo=UTC)


def test_live_archives_source_columns_without_calibration(monkeypatch, tmp_path):
    monkeypatch.setenv('ARGUS_GOES_ARCHIVE', str(tmp_path))
    frame = pd.DataFrame([{'timestamp': pd.Timestamp(NOW-timedelta(minutes=10)),
                           'goes_euv_256': 2.5, 'goes_euvs_quality_valid': True}])
    fetch = Mock(return_value=[frame])
    monkeypatch.setattr(goes, 'source_frames', fetch)
    records = goes.download_snapshots(NOW-timedelta(days=1), NOW, False)
    assert len(records) == 1
    row = records[0]
    samples = json.loads((tmp_path / row['raw_path']).read_text())
    assert samples[0]['goes_euv_256'] == 2.5
    assert not {'s10', 'm10', 'y10'} & samples[0].keys()
    monkeypatch.setattr(goes, 'source_frames', Mock(side_effect=AssertionError('Fetched retained snapshot')))
    assert goes.download_snapshots(NOW-timedelta(days=1), NOW, False,
                                   {(row['slot_at'], row['source_product']): row}) == records


def test_archive_preserves_daily_receipts_and_rejects_revised_recovery(monkeypatch, tmp_path):
    monkeypatch.setenv('ARGUS_GOES_ARCHIVE', str(tmp_path))
    end = NOW.replace(hour=0, minute=0)
    start = end-timedelta(days=2)
    frame = pd.DataFrame({'timestamp': pd.date_range(start, periods=48, freq='h'), 'goes_euv_256': 2.5})
    monkeypatch.setattr(goes, 'source_frames', lambda *_: [frame])
    records = goes.download_snapshots(start, end, True)
    assert len(records) == 2
    expected = {(r['slot_at'], r['source_product']): r for r in records}
    row = records[0]
    path = tmp_path / row['raw_path']
    path.unlink()
    path.with_suffix('.json.receipt.json').unlink()
    frame['goes_euv_256'] = 3.5
    with pytest.raises(ValueError, match='SHA256'):
        goes.download_snapshots(start, end, True, expected)
    assert not path.exists()
