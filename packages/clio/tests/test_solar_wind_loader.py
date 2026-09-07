from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from clio.dataloaders import solar_wind_loader as loader

NOW = datetime(2026, 9, 7, 12, tzinfo=UTC)


def mag(**overrides):
    return {"time_tag": "2026-09-07T11:59:00", "source": "SOLAR1", "active": True,
            "bx_gsm": -1.2, "by_gsm": 2.3, "bz_gsm": -4.5, "bt": 5.2,
            "overall_quality": 0, "max_data_flag": -9999, **overrides}


def test_native_samples_keep_spacecraft_selection_time_and_flags():
    result = loader.parse_records('mag', [mag(), mag(source='ACE', active=False)], NOW)
    assert len(result) == 2
    assert result[0]['observed_at'] == datetime(2026, 9, 7, 11, 59, tzinfo=UTC)
    assert result[0]['received_at'] == NOW
    assert result[0]['values'] == {'bx': -1.2, 'by': 2.3, 'bz': -4.5, 'bt': 5.2}
    assert result[0]['raw']['max_data_flag'] == -9999
    assert result[1]['spacecraft'] == 'ACE'
    assert result[1]['active'] is False


def test_gaps_are_not_filled_and_invalid_values_do_not_become_zero():
    result = loader.parse_records('mag', [mag(time_tag='2026-09-07T11:50:00'),
                                          mag(bx_gsm=-99999, by_gsm='NaN', bz_gsm=None, bt=-1)], NOW)
    assert len(result) == 2
    assert result[1]['values'] == dict.fromkeys(['bx', 'by', 'bz', 'bt'])


def test_plasma_fields_are_scalar_values_not_propagated():
    row = {'time_tag': '2026-09-07T11:59:00Z', 'source': 'SOLAR1', 'active': True,
           'proton_speed': '420.5', 'proton_density': '0', 'proton_temperature': -9999}
    assert loader.parse_records('plasma', [row], NOW)[0]['values'] == {'v': 420.5, 'n': 0, 't': None}


@pytest.mark.parametrize('payload', [None, [], [['time_tag']], [mag(active='true')], [mag(time_tag='bad')], [{}]])
def test_schema_failures_are_explicit(payload):
    with pytest.raises(ValueError):
        loader.parse_records('mag', payload, NOW)


def test_duplicate_records_use_last_revision_and_future_records_are_ignored():
    records = loader.parse_records('mag', [mag(), mag(bt=8), mag(time_tag='2026-09-08T12:00:00')], NOW)
    assert len(records) == 1
    assert records[0]['values']['bt'] == 8


def test_fetch_uses_current_rtsw_url_and_http_timeout(monkeypatch):
    response = Mock(json=Mock(return_value=[mag()]))
    get = Mock(return_value=response)
    monkeypatch.setattr(loader.requests, 'get', get)
    loader.fetch_records('mag')
    get.assert_called_once_with(loader.SOURCES['mag'], timeout=(5, 25))
    response.raise_for_status.assert_called_once()
