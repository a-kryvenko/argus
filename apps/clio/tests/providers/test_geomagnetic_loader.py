from datetime import UTC, datetime, timedelta
import pytest
from clio.providers.geomagnetic_loader import parse_records

NOW = datetime(2026, 9, 7, 16, tzinfo=UTC)


def test_fractional_kp_and_ap_preserve_native_time():
    row = {'time_tag': '2026-09-07T12:00:00', 'Kp': 3.33, 'a_running': '18', 'station_count': 8}
    result = parse_records('kp', [row], NOW)
    assert result.to_dict('records') == [dict(timestamp=NOW.replace(hour=12), kp=3.33, ap=18.)]


def test_dst_is_hourly_and_negative_values_are_preserved():
    result = parse_records('dst', [{'time_tag': '2026-09-07T15:00:00Z', 'dst': -123}], NOW)
    assert result.dst.tolist() == [-123.]
    assert result.timestamp.tolist() == [NOW.replace(hour=15)]


@pytest.mark.parametrize('value', [None, 'NaN', -99999, 9999, 12, -1, True])
def test_invalid_kp_is_rejected(value):
    assert parse_records('kp', [{'time_tag': '2026-09-07T12:00:00', 'Kp': value}], NOW).empty


def test_invalid_kp_keeps_independent_valid_ap():
    result = parse_records('kp', [{'time_tag': '2026-09-07T12:00:00', 'Kp': 12, 'a_running': 18}], NOW)
    assert result.kp.isna().all() and result.ap.tolist() == [18.]


def test_duplicate_revisions_and_missing_stations_are_rejected():
    row = {'time_tag': '2026-09-07T15:00:00', 'Kp': 3.33, 'station_count': 8}
    result = parse_records('kp', [row, {**row, 'Kp': 4.33}], NOW)
    assert result.kp.tolist() == [4.33]
    assert parse_records('kp', [row, {**row, 'station_count': 0, 'a_running': 18}], NOW).empty


@pytest.mark.parametrize('payload', [[], {}, [{}], [['time_tag', 'Kp']],
    [{'time_tag': 'bad', 'Kp': 3}], [{'time_tag': '2026-09-07T13:00:00Z', 'Kp': 3}]])
def test_schema_drift_is_not_silently_accepted(payload):
    with pytest.raises(ValueError):
        parse_records('kp', payload, NOW)
