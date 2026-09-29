"""Bounded periodic scoring, using the same exclusive writer as manual verify."""
from datetime import UTC, datetime, timedelta
from time import monotonic

from argus_prophet.db.session import connect
from argus_prophet.scheduling.jobs import verification_lock
from argus_prophet.scheduling.execution import before_dispatch


def run_due(config, *, now=None, verify=None):
    if not config.enabled:
        return False
    before_dispatch()
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    every = timedelta(hours=config.every_hours)
    slot = epoch + ((now-epoch)//every)*every
    with verification_lock():
        with connect(writing=True) as conn:
            previous = conn.execute("SELECT completed_slot FROM prophet.scheduled_job WHERE name='verification'").fetchone()
        if previous and previous[0] >= slot:
            return False
        if verify is None:
            from argus_prophet.services.verification import verify
        verify('all', days=config.days, now=now, deadline=monotonic()+config.timeout_seconds)
        with connect(writing=True) as conn:
            conn.execute('''INSERT INTO prophet.scheduled_job(name,completed_slot,completed_at)
                VALUES ('verification',%s,%s) ON CONFLICT (name) DO UPDATE SET
                completed_slot=EXCLUDED.completed_slot,completed_at=EXCLUDED.completed_at''',
                (slot, datetime.now(UTC)))
    return True
