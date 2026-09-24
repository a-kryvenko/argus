from datetime import UTC, datetime, timedelta
from argus_clio.services.coverage import coverage

START = datetime(2026, 9, 1, tzinfo=UTC)


def point(minutes, value=1, quality='unverified', intervals=False):
    return {('interval_start' if intervals else 'observed_at'): START+timedelta(minutes=minutes),
            'value': value, 'quality': quality}


def test_missing_invalid_and_merged_gaps():
    result = coverage([point(0), point(1, quality='flagged'), point(4, value=None)], START, START+timedelta(minutes=6), 60)
    assert (result['expected_slots'], result['usable_slots'], result['missing_slots'], result['invalid_slots']) == (6, 1, 3, 2)
    assert result['percent'] == 16.67
    assert [(gap['reason'], gap['slots']) for gap in result['gaps']] == [('invalid', 1), ('missing', 2), ('invalid', 1), ('missing', 1)]


def test_minute_boundaries_and_duplicate_samples():
    result = coverage([point(1), point(1), point(2)], START+timedelta(seconds=30), START+timedelta(minutes=2), 60)
    assert result['expected_slots'] == result['usable_slots'] == 1
    assert result['percent'] == 100 and result['gaps'] == []


def test_overlapping_native_intervals_are_counted_once():
    result = coverage([point(0, intervals=True)], START+timedelta(hours=1), START+timedelta(hours=4), 10800, intervals=True)
    assert result['expected_slots'] == 2 and result['percent'] == 50
    assert result['gaps'][0]['from'] == START+timedelta(hours=3)
    assert result['gaps'][0]['to'] == START+timedelta(hours=4)


def test_future_is_not_missing_history():
    result = coverage([point(0)], START, START+timedelta(days=1), 60, now=START+timedelta(minutes=1))
    assert result['expected_slots'] == result['usable_slots'] == 1
    empty = coverage([], START, START+timedelta(seconds=20), 60, now=START)
    assert empty['percent'] is None and empty['gaps'] == []


def test_empty_history_and_partial_interval():
    result = coverage([], START+timedelta(minutes=30), START+timedelta(hours=2), 3600, intervals=True)
    assert result['percent'] == 0 and result['missing_slots'] == 2
    assert result['gaps'] == [{'from': START+timedelta(minutes=30), 'to': START+timedelta(hours=2), 'reason': 'missing', 'slots': 2}]


def test_future_partial_interval_has_no_expected_slots():
    result = coverage([], START+timedelta(minutes=30), START+timedelta(hours=1), 3600,
                      intervals=True, now=START+timedelta(minutes=15))
    assert result['expected_slots'] == 0 and result['gaps'] == []
