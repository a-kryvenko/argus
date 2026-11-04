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
        from clio.ingestion.backfill_files import backfill_files
        report = await backfill_files(config, files, now=now)
        result.update({key: value for key, value in report.items() if key not in ('status', 'failed_metrics')})
        result['failed_metrics'] = sorted(set(result['failed_metrics']) | set(report['failed_metrics']))
        if report['status'] == 'partial':
            result['status'] = 'partial'
    return result
