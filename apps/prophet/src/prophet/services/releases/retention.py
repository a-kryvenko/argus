"""Expire diagnostic evidence independently of forecast and accuracy history."""
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from prophet.db.session import get_db_session
from prophet.db.locks import transaction


DETAIL_DAYS = 7
EXPIRED_DETAILS = """status <> 'running' AND started_at < :cutoff
    AND (input_snapshot IS NOT NULL OR input_sha256 IS NOT NULL
         OR provenance <> '{}'::jsonb OR error IS NOT NULL)"""


async def cleanup(*, apply=False, now=None, writer=None):
    now = now or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('now must include a timezone')
    cutoff = now - timedelta(days=DETAIL_DAYS)
    async with (transaction(writer) if apply else get_db_session()) as session:
        if apply:
            result = await session.execute(text(f"""UPDATE forecast_run
                SET input_snapshot=NULL,input_sha256=NULL,provenance='{{}}'::jsonb,error=NULL
                WHERE {EXPIRED_DETAILS}"""), {'cutoff': cutoff})
            count = result.rowcount
        else:
            count = await session.scalar(text(f'SELECT count(*) FROM forecast_run WHERE {EXPIRED_DETAILS}'),
                                         {'cutoff': cutoff})
    return {'applied': apply, 'before': cutoff.isoformat(), 'details': count}
