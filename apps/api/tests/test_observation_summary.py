from datetime import UTC, datetime, timedelta
import pytest
from app.services.observation_summary import change_one_hour, southward_duration

NOW = datetime(2026, 9, 7, 16, tzinfo=UTC)


def points(count=65):
    return [{'observed_at': NOW-timedelta(minutes=count-1-i), 'value': float(i),
             'quality': 'unverified', 'spacecraft': 'SOLAR1'} for i in range(count)]


def test_change_compares_five_minute_means_an_hour_apart():
    result = change_one_hour(points(), NOW)
    assert result['status'] == 'available' and result['value'] == 60
    assert result['recent_mean'] == 62 and result['previous_mean'] == 2
    assert result['coverage'] == {'recent': 1, 'previous': 1, 'last_hour': 1}


def test_coverage_requires_each_window_and_the_last_hour():
    rows = points()
    assert change_one_hour(rows[1:], NOW)['status'] == 'available'  # 80% baseline coverage
    assert change_one_hour(rows[2:], NOW)['reason'] == 'insufficient_coverage'
    assert change_one_hour(rows[:5]+rows[-5:], NOW)['reason'] == 'insufficient_coverage'


@pytest.mark.parametrize('method', [change_one_hour, southward_duration])
def test_stale_and_missing_latest_are_not_reported_as_current(method):
    rows = points()
    assert method(rows, NOW+timedelta(minutes=11))['reason'] == 'stale'
    rows[-1]['value'] = None
    assert method(rows, NOW)['reason'] == 'missing_latest'
    assert method([], NOW)['reason'] == 'missing_latest'


def test_source_switch_suppresses_trend():
    rows = points()
    rows[10]['spacecraft'] = 'ACE'
    assert change_one_hour(rows, NOW)['reason'] == 'source_changed'


def test_bz_duration_needs_consecutive_valid_samples_and_known_onset():
    rows = points(10)
    for row in rows[-5:]:
        row['value'] = -2
    result = southward_duration(rows, NOW)
    assert result['status'] == 'available' and result['value'] == 5
    assert southward_duration(rows[:-3]+rows[-2:], NOW)['reason'] == 'insufficient_coverage'
    rows[-3]['spacecraft'] = 'ACE'
    assert southward_duration(rows, NOW)['reason'] == 'source_changed'


def test_bz_never_infers_onset_from_window_start():
    rows = points(65)
    for row in rows:
        row['value'] = -1
    result = southward_duration(rows, NOW)
    assert result['status'] == 'lower_bound' and result['value'] == 65
    rows[-1]['value'] = 0
    assert southward_duration(rows, NOW)['value'] == 0


def test_flagged_samples_do_not_count_as_coverage_or_duration():
    rows = points()
    rows[0]['quality'] = rows[1]['quality'] = 'flagged'
    assert change_one_hour(rows, NOW)['reason'] == 'insufficient_coverage'
    rows[-1]['quality'] = 'flagged'
    assert southward_duration(rows, NOW)['reason'] == 'missing_latest'
