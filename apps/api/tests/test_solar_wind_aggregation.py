from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import pytest
from app.services.solar_wind_aggregation import summarize

START = datetime(2026, 9, 1, tzinfo=UTC)


def row(minute, value, source='A', active=True, flagged=False):
    return SimpleNamespace(observed_at=START+timedelta(minutes=minute), received_at=START+timedelta(hours=2),
        spacecraft=source, active=active, values={'bz': value}, raw={'overall_quality': int(flagged)})


def test_missing_flagged_extrema_and_negative_count():
    result = summarize([row(0, -8), row(1, 2), row(2, 99, flagged=True), row(4, -2, 'B')], 'mag', START, 300, START+timedelta(hours=1))
    bz = result['metrics']['bz']
    assert (bz['count'], bz['missing_count'], bz['invalid_count']) == (3, 1, 1)
    assert bz['sum'] == -8 and bz['min'] == -8 and bz['max'] == 2
    assert bz['negative_count'] == 2 and bz['coverage_percent'] == 60
    assert result['source_changes'] == 1 and result['max_gap_minutes'] == 1


def test_active_selection_and_tie_breaking():
    points = [row(0, -2, 'B'), row(0, -3, 'A'), row(1, 99, active=False)]
    assert summarize(points, 'mag', START, 300, START+timedelta(hours=1))['metrics']['bz']['sum'] == -3


def test_hour_mean_weights_individual_samples_and_excludes_end():
    points = [row(0, 10), *[row(i, 0) for i in range(5, 10)], row(60, 1000)]
    result = summarize(points, 'mag', START, 3600, START+timedelta(hours=1))
    assert result['metrics']['bz']['mean'] == 10/6
    assert result['window_complete']


def test_partial_and_empty_buckets():
    with pytest.raises(ValueError, match='closed windows'):
        summarize([], 'mag', START, 300, START+timedelta(minutes=2))
    result = summarize([], 'mag', START, 300, START+timedelta(minutes=5))
    assert result['window_complete']
    assert result['metrics']['bz']['mean'] is None
    assert result['metrics']['bz']['missing_count'] == 5
