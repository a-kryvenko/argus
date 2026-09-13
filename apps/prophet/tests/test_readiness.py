from datetime import UTC, datetime, timedelta

from common.schemas.forecast_inputs import ForecastInputs, SourceMeasurement
from common.schemas.observation import Observation, ObservationPoint
from argus_prophet.readiness import input_diagnostics

NOW = datetime(2026, 9, 13, 10, tzinfo=UTC)


def point(time=NOW, **changes):
    fields = dict(issue_time=time, bx=1, by=1, bz=1, v=400, n=5, t=100000,
                  kp=2, dst=-5, ap=4, f10_7=120)
    fields.update(changes)
    return ObservationPoint(**fields)


def inputs(points, measurements=None):
    return ForecastInputs(as_of=NOW, read_at=NOW, observations=Observation(points=points),
                          measurements=measurements or [])


def test_reports_age_gaps_and_sources_without_claiming_readiness():
    report = input_diagnostics(inputs(
        [point(NOW - timedelta(hours=5)), point(NOW - timedelta(hours=1))],
        [SourceMeasurement(metric='f10_7', value=120, observed_at=NOW-timedelta(days=5))]))
    assert report['normalized']['age_hours'] == 1
    assert report['normalized']['max_gap_hours'] == 4
    assert report['density_sources']['f10_7']['age_hours'] == 120
    assert report['density_sources']['ap']['age_hours'] is None
    assert report['thresholds_configured'] is False
    assert report['normalized']['source_freshness_known'] is False
    assert report['normalized']['invalid_value_counts']['s10'] == 2
    assert report['issues'] == []


def test_future_duplicate_naive_and_nonfinite_rows_are_distinguished():
    report = input_diagnostics(inputs([point(), point(), point(NOW+timedelta(hours=1)),
                                       point(NOW.replace(tzinfo=None), v=float('nan'))]))
    assert set(report['issues']) == {'duplicate_normalized_timestamps', 'future_normalized_timestamps',
                                     'naive_normalized_timestamps', 'missing_or_nonfinite_required_values'}
    assert report['normalized']['age_hours'] == -1
    assert report['normalized']['naive_timestamp_count'] == 1


def test_empty_inputs_are_not_reported_as_healthy():
    report = input_diagnostics(inputs([]))
    assert report['normalized']['age_hours'] is None
    assert report['issues'] == ['no_aware_normalized_timestamps']


def test_existing_density_age_boundary_and_unconfigured_products():
    from argus_prophet.readiness import classify_freshness
    assert classify_freshness(None, NOW, 6) == 'unavailable'
    assert classify_freshness(NOW - timedelta(hours=6), NOW, 6) == 'within_age_limit'
    assert classify_freshness(NOW - timedelta(hours=6, seconds=1), NOW, 6) == 'stale'
    assert classify_freshness(NOW - timedelta(days=30), NOW) == 'unconfigured'
    assert classify_freshness(NOW + timedelta(seconds=1), NOW) == 'future_issue_time'
