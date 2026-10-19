from datetime import UTC, datetime, timedelta
import pytest
from clio.observations.summary import change_one_hour, interval_change

NOW = datetime(2026, 9, 7, 16, tzinfo=UTC)


def points(count=65):
    return [{'observed_at': NOW-timedelta(minutes=count-1-i), 'value': float(i),
             'quality': 'unverified', 'spacecraft': 'SOLAR1'} for i in range(count)]


def test_change_compares_five_minute_means_an_hour_apart():
    result = change_one_hour(points(), NOW)
    assert result['status'] == 'available' and result['value'] == 60
    assert result['recent_mean'] == 62 and result['previous_mean'] == 2
    assert result['coverage'] == {'recent': 1, 'previous': 1}


def test_change_only_requires_one_usable_sample_in_each_window():
    rows = points()
    result = change_one_hour([rows[4], rows[-1]], NOW)
    assert result['status'] == 'available' and result['value'] == 60
    assert change_one_hour(rows[5:], NOW)['reason'] == 'missing_comparison'
    assert change_one_hour(rows[:-5], NOW)['reason'] == 'missing_comparison'


def test_change_does_not_require_measurements_between_comparison_windows():
    rows = points()
    result = change_one_hour(rows[1:5]+rows[-4:], NOW)
    assert result['status'] == 'available'
    assert result['value'] == 60
    assert result['coverage'] == {'recent': 0.8, 'previous': 0.8}


def test_stale_and_missing_latest_are_not_reported_as_current():
    rows = points()
    assert change_one_hour(rows, NOW+timedelta(minutes=11))['reason'] == 'stale'
    rows[-1]['value'] = None
    assert change_one_hour(rows, NOW)['reason'] == 'missing_latest'
    assert change_one_hour([], NOW)['reason'] == 'missing_latest'


def test_source_switch_suppresses_trend():
    rows = points()
    rows[10]['spacecraft'] = 'ACE'
    assert change_one_hour(rows, NOW)['reason'] == 'source_changed'


def test_flagged_samples_do_not_count_as_coverage():
    rows = points()
    rows[0]['quality'] = rows[1]['quality'] = 'flagged'
    assert change_one_hour(rows, NOW)['value'] == 59
    rows[2]['quality'] = rows[3]['quality'] = rows[4]['quality'] = 'flagged'
    assert change_one_hour(rows, NOW)['reason'] == 'missing_comparison'
    rows[-1]['quality'] = 'flagged'
    assert change_one_hour(rows, NOW)['reason'] == 'missing_latest'


@pytest.mark.parametrize('start_minutes,expected', [(0, 2), (-30, 2), (-120, 0)])
def test_interval_change_uses_values_available_now_and_one_hour_ago(start_minutes, expected):
    rows = [
        {'interval_start': NOW-timedelta(hours=3), 'interval_end': NOW, 'value': 2, 'quality': None},
        {'interval_start': NOW+timedelta(minutes=start_minutes), 'interval_end': NOW+timedelta(hours=3),
         'value': 4, 'quality': 'unverified'},
    ]
    assert interval_change(rows, NOW)['value'] == expected


def test_summary_includes_all_wind_variables_and_geomagnetic_changes(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock
    from clio.observations import summary as service
    from clio.domains.solar_wind.observations import METADATA

    wind = {metric: {'points': points(), 'unit': 'unit'} for metric in METADATA}
    wind_history = AsyncMock(return_value={'series': wind})
    monkeypatch.setattr(service.solar_wind, 'latest', AsyncMock(return_value={'series': {}}))
    monkeypatch.setattr(service.solar_wind, 'history', wind_history)
    monkeypatch.setattr(service.geomagnetic, 'latest', AsyncMock(return_value={
        'series': {metric: {'status': 'fresh'} for metric in ('kp', 'dst')}}))
    monkeypatch.setattr(service.geomagnetic, 'history', AsyncMock(return_value={'series': {
        metric: {'unit': 'unit', 'points': [
            {'interval_start': NOW-timedelta(hours=1), 'interval_end': NOW, 'value': 1, 'quality': None},
            {'interval_start': NOW, 'interval_end': NOW+timedelta(hours=1), 'value': 3, 'quality': None},
        ]} for metric in ('kp', 'dst')
    }}))
    result = asyncio.run(service.summary(object(), NOW))
    assert 'southward_bz' not in result
    assert set(result['changes_1h']) == {*METADATA, 'kp', 'dst'}
    assert all(result['changes_1h'][metric]['value'] == 60 for metric in METADATA)
    assert result['changes_1h']['kp']['value'] == result['changes_1h']['dst']['value'] == 2
    assert wind_history.await_args.args[1] == list(METADATA)
