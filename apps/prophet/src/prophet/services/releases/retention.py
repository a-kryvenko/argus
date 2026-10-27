"""Explicit, bounded cleanup; current releases and latest attempts survive."""
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from prophet.db.session import get_db_session
from prophet.db.locks import transaction


async def cleanup(*, days=90, apply=False, now=None, writer=None):
    if not isinstance(days, int) or days < 26:
        raise ValueError('Retain at least 26 days, covering the verification window')
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    cutoff = now-timedelta(days=days)
    async with (transaction(writer) if apply else get_db_session()) as session:
        if not apply:
            await session.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'))
        ids = [row[0] for row in (await session.execute(text('''SELECT f.id FROM prophet.forecast_run f
            WHERE f.status <> 'running' AND f.finished_at < :cutoff
            AND EXISTS (SELECT 1 FROM prophet.forecast_run newer
                        WHERE newer.product=f.product AND (newer.started_at,newer.id) > (f.started_at,f.id))
            AND NOT EXISTS (SELECT 1 FROM prophet.forecast_release r
                JOIN prophet.current_forecast c ON c.release_id=r.id WHERE r.run_id=f.id)
            ORDER BY f.finished_at,f.id LIMIT 1000'''), {'cutoff': cutoff})).all()]
        counts = {'runs': len(ids), 'releases': 0, 'artifacts': 0, 'verifications': 0}
        for table, key, predicate in (
            ('forecast_release', 'releases', 'run_id=ANY(:ids)'),
            ('forecast_artifact', 'artifacts', 'run_id=ANY(:ids)'),
            ('forecast_verification', 'verifications',
             'release_id IN (SELECT id FROM prophet.forecast_release WHERE run_id=ANY(:ids))'),
        ):
            counts[key] = (await session.execute(text(f'SELECT count(*) FROM prophet.{table} WHERE {predicate}'), {'ids': ids})).first()[0]
        if apply and ids:
            await session.execute(text('''DELETE FROM prophet.forecast_verification WHERE release_id IN
                (SELECT id FROM prophet.forecast_release WHERE run_id=ANY(:ids))'''), {'ids': ids})
            await session.execute(text('DELETE FROM prophet.forecast_release WHERE run_id=ANY(:ids)'), {'ids': ids})
            await session.execute(text('DELETE FROM prophet.forecast_artifact WHERE run_id=ANY(:ids)'), {'ids': ids})
            await session.execute(text('DELETE FROM prophet.forecast_run WHERE id=ANY(:ids)'), {'ids': ids})
    return {'applied': apply, 'before': cutoff.isoformat(), 'batch_limit': 1000, **counts}
