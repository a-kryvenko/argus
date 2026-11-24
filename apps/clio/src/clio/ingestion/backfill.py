"""Backfill configured observations without replacing existing measurements."""
from clio.db.session import get_db_session
from clio.ingestion.utils.products import OBSERVATIONS


async def backfill(config, *, now):
    metrics = [metric for metric, policy in config.observations.items() if policy.sources.historical]
    numeric = [metric for metric in metrics if OBSERVATIONS[metric].kind == 'numeric']
    files = [metric for metric in metrics if OBSERVATIONS[metric].kind == 'file']
    result = {'status': 'complete', 'failed_metrics': [], 'downloaded_measurements': 0}
    if numeric:
        from clio.ingestion.backfill_numeric import backfill_numeric
        async with get_db_session() as session:
            result = await backfill_numeric(session, config, numeric, now=now, raise_on_failure=False)
    if files:
        from clio.ingestion.collect_files import collect_files
        report = await collect_files(config, files, mode='backfill', now=now)
        result['files'] = report['files']
        result['failed_metrics'] = sorted(set(result['failed_metrics']) | set(report['failed_metrics']))
        if report['status'] == 'partial':
            result['status'] = 'partial'
        elif report['status'] == 'busy' and result['status'] == 'complete':
            result['status'] = 'busy'
    return result
