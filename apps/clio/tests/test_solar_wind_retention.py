from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import asyncio
import pytest
from argus_clio.services.solar_wind.aggregation import summarize, VERSION
from argus_clio.services.solar_wind.retention import verify_hour, cleanup

START = datetime(2026, 1, 1, tzinfo=UTC)
NOW = START+timedelta(days=100)


def test_all_thirteen_buckets_and_current_values_are_required():
    raw = [SimpleNamespace(observed_at=START, received_at=START, spacecraft='A', active=True,
                           values={'bz': -4}, raw={})]
    saved = [SimpleNamespace(resolution_seconds=seconds, bucket_start=START+timedelta(seconds=offset),
             version=VERSION, statistics=summarize(raw, 'mag', START+timedelta(seconds=offset), seconds, NOW))
             for seconds in (300, 3600) for offset in range(0, 3600, seconds)]
    assert verify_hour(raw, saved, 'mag', START, NOW) is None
    assert verify_hour(raw, saved[:-1], 'mag', START, NOW)['reason'] == 'aggregate_missing'
    raw[0].values['bz'] = -8
    issue = verify_hour(raw, saved, 'mag', START, NOW)
    assert issue['reason'] == 'aggregate_mismatch' and 'metrics.bz.min' in issue['fields']


@pytest.mark.parametrize('options', [{'retention_days': 89}, {'limit': 0}, {'limit': 241}])
def test_unsafe_limits_rejected_before_database_access(options):
    with pytest.raises(ValueError):
        asyncio.run(cleanup(**options))
