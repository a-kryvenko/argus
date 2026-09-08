from datetime import UTC, datetime, timedelta
import pytest
from clio.dataloaders.geomagnetic_loader import parse_records

NOW = datetime(2026, 9, 7, 16, tzinfo=UTC)


def test_fractional_kp_and_native_three_hour_interval():
    row = {'time_tag': '2026-09-07T12:00:00', 'Kp': 3.33, 'station_count': 8}
    point = parse_records('kp', [row], NOW)[0]
    assert point['value'] == 3.33
    assert point['interval_end']-point['interval_start'] == timedelta(hours=3)
    assert point['received_at'] == NOW
    assert point['raw'] == row
    assert point['quality'] == 'unverified'


def test_dst_is_hourly_and_negative_values_are_preserved():
    point = parse_records('dst', [{'time_tag': '2026-09-07T15:00:00Z', 'dst': -123}], NOW)[0]
    assert point['value'] == -123
    assert point['interval_end'] == NOW


@pytest.mark.parametrize('value', [None, 'NaN', -99999, 9999, 12, -1, True])
def test_invalid_kp_is_null_not_zero(value):
    point = parse_records('kp', [{'time_tag': '2026-09-07T12:00:00', 'Kp': value}], NOW)[0]
    assert point['value'] is None and point['quality'] == 'missing'


def test_duplicate_revisions_missing_stations_and_in_progress_intervals():
    row = {'time_tag': '2026-09-07T15:00:00', 'Kp': 3.33, 'station_count': 0}
    records = parse_records('kp', [row, {**row, 'Kp': 4.33}], NOW)
    assert len(records) == 1
    assert records[0]['value'] == 4.33 and records[0]['quality'] == 'flagged'
    assert records[0]['interval_end'] > NOW


@pytest.mark.parametrize('payload', [[], {}, [{}], [['time_tag', 'Kp']],
    [{'time_tag': 'bad', 'Kp': 3}], [{'time_tag': '2026-09-07T13:00:00Z', 'Kp': 3}]])
def test_schema_drift_is_not_silently_accepted(payload):
    with pytest.raises(ValueError):
        parse_records('kp', payload, NOW)
