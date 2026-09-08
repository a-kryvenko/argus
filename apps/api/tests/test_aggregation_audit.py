from datetime import UTC, datetime
import argparse
import pytest

from app.commands.audit_solar_wind import utc_hour
from app.services.aggregation_audit import differences


def test_comparison_tolerates_only_float_noise_and_reports_field_paths():
    expected = {'metrics': {'bz': {'count': 5, 'mean': 1/3, 'min': -8, 'max': 2, 'coverage_percent': 100.0}}}
    actual = {'metrics': {'bz': {'count': 4, 'mean': 1/3+1e-12, 'min': -8, 'max': 3, 'coverage_percent': 80.0}}}
    assert differences(expected, actual) == ['metrics.bz.count', 'metrics.bz.coverage_percent', 'metrics.bz.max']
    assert differences({'mean': None}, {'mean': 0}) == ['mean']
    assert differences({'source_sequence': [{'spacecraft': 'A'}]}, {'source_sequence': []}) == ['source_sequence']
    assert differences({'a': 1}, {}) == ['a']


def test_cli_requires_timezone_and_hour_alignment():
    assert utc_hour('2026-09-01T02:00:00+02:00') == datetime(2026, 9, 1, tzinfo=UTC)
    for value in ('2026-09-01T00:00:00', '2026-09-01T00:30:00Z', 'invalid'):
        with pytest.raises(argparse.ArgumentTypeError):
            utc_hour(value)
