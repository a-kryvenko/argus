from datetime import UTC, datetime
from clio.scheduling.jobs import slot_for


def test_calendar_slots_are_aligned():
    now = datetime(2026, 9, 12, 12, 19, 45, tzinfo=UTC)
    assert slot_for('aggregate', now) == now.replace(minute=15, second=0)
    assert slot_for('refresh', now) == now.replace(minute=0, second=0)


def test_source_and_job_locks_never_overlap():
    from clio.db.locks import SOURCE_LOCKS, JOB_LOCKS
    from clio.monitoring.specs import SOURCE_SPECS
    from clio.scheduling.jobs import JOBS

    assert set(SOURCE_LOCKS) == set(SOURCE_SPECS)
    assert set(JOB_LOCKS) == set(JOBS) | {'live', 'aia-live', 'aia'}
    keys = [*SOURCE_LOCKS.values(), *JOB_LOCKS.values()]
    assert len(set(keys)) == len(keys)
