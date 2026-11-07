"""Collect configured live observations."""
from clio.ingestion.utils.products import OBSERVATIONS


async def collect(config, metrics=None, *, now):
    extras = {'donki'}
    selected = list(config.observations) if metrics is None else list(dict.fromkeys(metrics))
    if any(metric not in config.observations and metric not in extras for metric in selected):
        raise ValueError('Select observations configured in clio.observations')
    numeric = [metric for metric in selected if metric not in extras and OBSERVATIONS[metric].kind == 'numeric' and config.observations[metric].sources.live]
    files = [metric for metric in selected if metric in extras or OBSERVATIONS[metric].kind == 'file' and config.observations[metric].sources.live]
    result = {'status': 'complete', 'failed_metrics': [], 'downloaded_measurements': 0}
    if numeric:
        from clio.ingestion.collect_numeric import collect_numeric
        result = await collect_numeric(config, numeric, now=now)
    if files or metrics is None:
        from clio.ingestion.collect_files import collect_files
        report = await collect_files(config, None if metrics is None else files, now=now)
        result.update({key: value for key, value in report.items() if key not in ('status', 'failed_metrics')})
        result['failed_metrics'] = sorted(set(result['failed_metrics']) | set(report['failed_metrics']))
        if report['status'] == 'partial':
            result['status'] = 'partial'
    return result
