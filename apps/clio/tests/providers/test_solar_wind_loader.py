from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from clio.providers import solar_wind as loader

NOW = datetime(2026, 9, 7, 12, tzinfo=UTC)


def mag(**overrides):
    return {"time_tag": "2026-09-07T11:59:00", "source": "SOLAR1", "active": True,
            "bx_gsm": -1.2, "by_gsm": 2.3, "bz_gsm": -4.5, "bt": 5.2,
            "overall_quality": 0, "max_data_flag": -9999, **overrides}


def test_native_samples_select_active_spacecraft_without_metadata():
    result = loader.parse_records('mag', [mag(), mag(source='ACE', active=False)], NOW)
    assert result.to_dict('records') == [dict(timestamp=NOW.replace(minute=59, hour=11),
        bx=-1.2, by=2.3, bz=-4.5, bt=5.2)]


def test_gaps_are_not_filled_and_invalid_values_do_not_become_zero():
    result = loader.parse_records('mag', [mag(time_tag='2026-09-07T11:50:00'),
        mag(bx_gsm=-99999, by_gsm='NaN', bz_gsm=None, bt=-1)], NOW)
    assert result.timestamp.tolist() == [NOW.replace(hour=11, minute=50)]


def test_plasma_filters_individual_values_without_losing_valid_zero():
    row = {'time_tag': '2026-09-07T11:59:00Z', 'source': 'SOLAR1', 'active': True,
           'proton_speed': '420.5', 'proton_density': '0', 'proton_temperature': -9999}
    result = loader.parse_records('plasma', [row], NOW)
    assert result.v.tolist() == [420.5] and result.n.tolist() == [0.]
    assert result.t.isna().all()


def test_flagged_and_inactive_records_are_rejected():
    assert loader.parse_records('mag', [mag(overall_quality=1), mag(active=False)], NOW).empty


def test_invalid_revision_does_not_resurrect_older_value():
    assert loader.parse_records('mag', [mag(), mag(overall_quality=1)], NOW).empty


@pytest.mark.parametrize('payload', [None, [], [['time_tag']], [mag(active='true')], [mag(time_tag='bad')], [{}]])
def test_schema_failures_are_explicit(payload):
    with pytest.raises(ValueError):
        loader.parse_records('mag', payload, NOW)


def test_duplicate_records_use_last_revision_and_future_records_are_ignored():
    records = loader.parse_records('mag', [mag(), mag(bt=8), mag(time_tag='2026-09-08T12:00:00')], NOW)
    assert len(records) == 1
    assert records.bt.iloc[0] == 8


def test_fetch_uses_current_rtsw_url_and_http_timeout(monkeypatch):
    response = Mock(json=Mock(return_value=[mag()]))
    get = Mock(return_value=response)
    monkeypatch.setattr(loader.requests, 'get', get)
    loader.fetch_records('mag')
    get.assert_called_once_with(loader.SOURCES['mag'], timeout=(5, 25))
    response.raise_for_status.assert_called_once()
