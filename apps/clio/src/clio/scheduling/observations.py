"""Batch due live/backfill observations with independent completion markers."""
import logging
from datetime import UTC, datetime, timedelta

from clio.db.locks import JOB_LOCKS
from clio.db.session import get_database_url
from clio.scheduling.jobs import JobBusy, work_cycles
from clio.ingestion.products import OBSERVATIONS

logger = logging.getLogger(__name__)
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def observation_slot(now: datetime, every: timedelta) -> datetime:
    """UTC epoch alignment supports intervals longer than an hour."""
    return EPOCH + ((now.astimezone(UTC) - EPOCH) // every) * every


def execute(config, run, *, now=None, mode='backfill', force=False):
    """Run one batch, checkpointing only metrics without unresolved failures."""
    if mode not in ('live', 'backfill'):
        raise ValueError('Unknown observation schedule')
    if not config.observations:
        return False
    import psycopg
    now = now or datetime.now(UTC)
    url = get_database_url()
    with psycopg.connect(url.set(drivername='postgresql').render_as_string(hide_password=False), autocommit=True,
                         options='-csearch_path=clio,pg_catalog,pg_temp') as conn:
        files_only = all(OBSERVATIONS[m].kind == 'file' for m in config.observations)
        job = ('aia-live' if mode == 'live' else 'aia') if files_only else ('live' if mode == 'live' else 'refresh')
        key = JOB_LOCKS[job]
        if not conn.execute('SELECT pg_try_advisory_lock(%s)', (key,)).fetchone()[0]:
            raise JobBusy(f'Clio {mode} is already running')
        slots = {metric: observation_slot(now, getattr(policy.schedules, mode).every)
                 for metric, policy in config.observations.items()}
        names = [f'{mode}.{metric}' for metric in slots]
        previous = dict(conn.execute(
            'SELECT name, completed_slot FROM clio.scheduled_job WHERE name = ANY(%s)', (names,)).fetchall())
        due = [metric for metric, slot in slots.items()
               if force or previous.get(f'{mode}.{metric}', EPOCH) < slot]
        if not due:
            return False
        result = run(due, now)
        failed = set(result['failed_metrics'])
        logger.info('Clio %s metrics=%s status=%s received=%s missing=%s files=%s',
                    mode, ','.join(due), result.get('status', 'unknown'),
                    result.get('downloaded_measurements', 0),
                    result.get('missing_observed_slots', result.get('missing_live_slots', {})),
                    {m: {k: v for k, v in report.items() if k in
                         ('received', 'restored', 'retained', 'missing', 'rejected', 'failed')}
                     for m, report in result.get('files', {}).items()})
        for attempt in result.get('source_attempts', []):
            if 'error' in attempt:
                logger.warning('Clio %s source=%s metrics=%s error=%s',
                               mode, attempt['product'], attempt.get('metrics'), attempt['error'])
        with conn.transaction():
            if failed:
                # A forced live startup can fail after a previous process marked
                # this slot complete. Invalidate it so the next cycle retries.
                conn.execute('DELETE FROM clio.scheduled_job WHERE name = ANY(%s)',
                             ([f'{mode}.{m}' for m in failed if m in due],))
            for metric in due:
                if metric in failed:
                    continue
                conn.execute('''INSERT INTO clio.scheduled_job(name, completed_slot, completed_at)
                    VALUES (%s, %s, %s) ON CONFLICT (name) DO UPDATE SET
                    completed_slot=EXCLUDED.completed_slot, completed_at=EXCLUDED.completed_at''',
                             (f'{mode}.{metric}', slots[metric], datetime.now(UTC)))
        if failed:
            logger.warning('%s will retry failed observations: %s', mode, ', '.join(sorted(failed)))
        return True


def work(config, run, *, mode='backfill'):
    poll_seconds = min([60, *(getattr(policy.schedules, mode).every.total_seconds()
                             for policy in config.observations.values())])
    first = mode == 'live'
    def cycle():
        nonlocal first
        result = execute(config, run, mode=mode, force=first)
        first = False
        return result
    work_cycles(mode, cycle, poll_seconds=poll_seconds)
