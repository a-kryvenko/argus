from unittest.mock import Mock, patch

import pandas as pd
import pytest

from clio.providers.omniweb_loader import OMNIWeb_Loader


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
    with patch('clio.providers.omniweb_loader.requests.post', return_value=response):
        result = OMNIWeb_Loader.load(pd.Timestamp('2025-12-31', tz='UTC'),
                                    pd.Timestamp('2025-12-31', tz='UTC'))
    assert len(result) == 1
    assert result.issue_time.iloc[0].hour == 23


def test_missing_records_report_provider_message():
    with pytest.raises(RuntimeError, match='Requested date outside available range'):
        OMNIWeb_Loader._parse_omni_text('<html><b>Requested date outside available range</b></html>')


def test_ace_http_error_is_not_reported_as_missing():
    import requests
    from clio.providers.spdf_loader import SPDF_Loader
    response = Mock(status_code=503)
    response.raise_for_status.side_effect = requests.HTTPError('503 unavailable')
    with patch('clio.providers.spdf_loader.requests.get', return_value=response):
        with pytest.raises(requests.HTTPError):
            SPDF_Loader._load_cdf_from_url('https://example.org/test.cdf')


def test_unpublished_range_is_empty_not_source_failure():
    response = Mock(text='Error INVALID START DATE, correct range: 19631128 - 20260916')
    with patch('clio.providers.omniweb_loader.requests.post', return_value=response) as post:
        result = OMNIWeb_Loader.load(pd.Timestamp('2026-09-22', tz='UTC'),
                                    pd.Timestamp('2026-09-22', tz='UTC'))
    assert result.empty
    assert 'v' in result and 'issue_time' in result
    assert post.call_count == 1


def test_partial_coverage_retries_only_available_dates():
    responses = [Mock(text='Error INVALID END DATE, correct range: 19631128 - 20260916'),
                 Mock(text='2026 259 23 1 2 -3 10000 5 400 27 -20 12 123.4')]
    calls = []
    def post(url, data, **kwargs):
        calls.append(dict(data))
        return responses.pop(0)
    with patch('clio.providers.omniweb_loader.requests.post', side_effect=post):
        result = OMNIWeb_Loader.load(pd.Timestamp('2026-09-15', tz='UTC'),
                                    pd.Timestamp('2026-09-22', tz='UTC'))
    assert calls[1]['start_date'] == '20260915'
    assert calls[1]['end_date'] == '20260916'
    assert len(result) == 1 and result.v.iloc[0] == 400


def test_unrecognized_error_still_fails():
    with patch('clio.providers.omniweb_loader.requests.post', return_value=Mock(text='Service unavailable')):
        with pytest.raises(RuntimeError, match='Service unavailable'):
            OMNIWeb_Loader.load(pd.Timestamp('2026-09-22', tz='UTC'),
                                pd.Timestamp('2026-09-22', tz='UTC'))
