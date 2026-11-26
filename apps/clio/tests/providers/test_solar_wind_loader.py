from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from clio.providers import solar_wind as loader

NOW = datetime(2026, 9, 7, 12, tzinfo=UTC)


def mag(**overrides):
    return {"time_tag": "2026-09-07T11:59:00", "source": "SOLAR1", "active": True,
            "bx_gsm": -1.2, "by_gsm": 2.3, "bz_gsm": -4.5, "bt": 5.2,
            "overall_quality": 0, "max_data_flag": -9999, **overrides}


def test_native_samples_keep_selected_active_spacecraft():
    result = loader.parse_records('mag', [mag(), mag(source='ACE', active=False)], NOW)
    assert result.set_index('metric').value.to_dict() == {'bx': -1.2, 'by': 2.3, 'bz': -4.5, 'bt': 5.2}
    assert result.observed_at.eq(NOW.replace(minute=59, hour=11)).all()
    assert result.spacecraft.eq('SOLAR1').all()


def test_gaps_are_not_filled_and_invalid_values_do_not_become_zero():
    result = loader.parse_records('mag', [mag(time_tag='2026-09-07T11:50:00'),
        mag(bx_gsm=-99999, by_gsm='NaN', bz_gsm=None, bt=-1)], NOW)
    assert set(result.loc[result.value.notna(), 'observed_at']) == {NOW.replace(hour=11, minute=50)}
    assert len(result.loc[result.quality == 'flagged']) == 3
    assert len(result.loc[result.quality == 'missing']) == 1


def test_plasma_filters_individual_values_without_losing_valid_zero():
    row = {'time_tag': '2026-09-07T11:59:00Z', 'source': 'SOLAR1', 'active': True,
           'proton_speed': '420.5', 'proton_density': '0', 'proton_temperature': -9999}
    result = loader.parse_records('plasma', [row], NOW)
    good = result.loc[result.quality.isna()]
    assert good.set_index('metric').value.to_dict() == {'v': 420.5, 'n': 0.}
    assert result.loc[result.metric == 't', 'value'].isna().all()


def test_inactive_records_are_not_selected():
    assert loader.parse_records('mag', [mag(active=False)], NOW).empty


def test_invalid_revision_does_not_resurrect_older_value():
    flagged = loader.parse_records('mag', [mag(), mag(overall_quality=1)], NOW)
    assert len(flagged) == 4 and flagged.quality.eq('flagged').all() and flagged.value.isna().all()


@pytest.mark.parametrize('payload', [None, [], [['time_tag']], [mag(active='true')], [mag(time_tag='bad')], [{}]])
def test_schema_failures_are_explicit(payload):
    with pytest.raises(ValueError):
        loader.parse_records('mag', payload, NOW)


def test_duplicate_records_use_last_revision_and_future_records_are_ignored():
    records = loader.parse_records('mag', [mag(), mag(bt=8), mag(time_tag='2026-09-08T12:00:00')], NOW)
    assert records.observed_at.nunique() == 1
    assert records.loc[records.metric == 'bt', 'value'].iloc[0] == 8


def test_fetch_uses_current_rtsw_url_and_http_timeout(monkeypatch):
    response = Mock(json=Mock(return_value=[mag()]))
    get = Mock(return_value=response)
    monkeypatch.setattr(loader.requests, 'get', get)
    loader.fetch_records('mag')
    get.assert_called_once_with(loader.SOURCES['mag'], timeout=(5, 25))
    response.raise_for_status.assert_called_once()
