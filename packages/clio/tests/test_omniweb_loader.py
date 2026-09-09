from unittest.mock import Mock, patch

import pandas as pd
import pytest

from clio.dataloaders.omniweb_loader import OMNIWeb_Loader


def test_column_specific_fill_and_fractional_kp():
    result = OMNIWeb_Loader._parse_omni_text(
        '2025 001 00 999.9 999.9 999.9 9999999 999.9 9999 99 99999 999 999.9\n'
        '2025 001 01 1 2 -3 10000 5 400 27 -20 12 123.4')
    assert result.iloc[0].drop('issue_time').isna().all()
    row = result.iloc[1]
    assert row.kp == pytest.approx(2.7)
    assert row.t == 10000  # Real temperature, not the old generic fill mask.
    assert row.f10_7 == 123.4


def test_requested_end_day_includes_last_hour():
    response = Mock(text='2025 365 23 1 2 -3 10000 5 400 27 -20 12 123.4')
    with patch('clio.dataloaders.omniweb_loader.requests.post', return_value=response):
        result = OMNIWeb_Loader.load(pd.Timestamp('2025-12-31', tz='UTC'),
                                    pd.Timestamp('2025-12-31', tz='UTC'))
    assert len(result) == 1
    assert result.issue_time.iloc[0].hour == 23
