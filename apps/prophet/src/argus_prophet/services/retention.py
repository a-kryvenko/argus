"""Explicit, bounded cleanup; current releases and latest attempts survive."""
from datetime import UTC, datetime, timedelta

from argus_prophet.db.session import connect


def cleanup(*, days=90, apply=False, now=None):
    if not isinstance(days, int) or days < 26:
        raise ValueError('Retain at least 26 days, covering the verification window')
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    cutoff = now-timedelta(days=days)
    with connect(writing=apply) as conn:
        if not apply:
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        ids = [row[0] for row in conn.execute('''SELECT f.id FROM prophet.forecast_run f
            WHERE f.status <> 'running' AND f.finished_at < %s
            AND EXISTS (SELECT 1 FROM prophet.forecast_run newer
                        WHERE newer.product=f.product AND (newer.started_at,newer.id) > (f.started_at,f.id))
            AND NOT EXISTS (SELECT 1 FROM prophet.forecast_release r
                JOIN prophet.current_forecast c ON c.release_id=r.id WHERE r.run_id=f.id)
            ORDER BY f.finished_at,f.id LIMIT 1000''', (cutoff,)).fetchall()]
        counts = {'runs': len(ids), 'releases': 0, 'artifacts': 0, 'verifications': 0}
        for table, key, predicate in (
            ('forecast_release', 'releases', 'run_id=ANY(%s)'),
            ('forecast_artifact', 'artifacts', 'run_id=ANY(%s)'),
            ('forecast_verification', 'verifications',
             'release_id IN (SELECT id FROM prophet.forecast_release WHERE run_id=ANY(%s))'),
        ):
            counts[key] = conn.execute(f'SELECT count(*) FROM prophet.{table} WHERE {predicate}', (ids,)).fetchone()[0]
        if apply and ids:
            conn.execute('''DELETE FROM prophet.forecast_verification WHERE release_id IN
                (SELECT id FROM prophet.forecast_release WHERE run_id=ANY(%s))''', (ids,))
            conn.execute('DELETE FROM prophet.forecast_release WHERE run_id=ANY(%s)', (ids,))
            conn.execute('DELETE FROM prophet.forecast_artifact WHERE run_id=ANY(%s)', (ids,))
            conn.execute('DELETE FROM prophet.forecast_run WHERE id=ANY(%s)', (ids,))
    # Slots are small durable scheduling evidence; retaining them preserves the
    # scheduler's completed-hour watermark even across clock rollback.
    return {'applied': apply, 'before': cutoff.isoformat(), 'batch_limit': 1000, **counts}
