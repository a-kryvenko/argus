"""Backfill configured observations without changing existing records."""

from clio.db.session import get_db_session
from clio.ingestion.selection import partition, merge_files, merge_sdo


async def backfill(config, *, now):
    numeric, files = partition(config, None)
    result = {'status': 'complete', 'failed_metrics': [], 'downloaded_measurements': 0}
    if numeric:
        from clio.ingestion.numeric_backfill import backfill_numeric
        async with get_db_session() as session:
            result = await backfill_numeric(session, config, numeric, now=now,
                                             raise_on_failure=False)
    if files:
        from clio.ingestion.files import collect_file_observations
        result = merge_files(result, await collect_file_observations(
            config, files, mode='backfill', now=now))
    result = await merge_sdo(result, config, now=now, mode='warmup')
    return result
