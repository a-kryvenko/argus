"""Collect configured live observations."""
from clio.ingestion.utils.products import OBSERVATIONS


async def collect(config, metrics=None, *, now):
    selected = list(config.observations) if metrics is None else list(dict.fromkeys(metrics))
    if any(metric not in config.observations for metric in selected):
        raise ValueError('Select observations configured in clio.observations')
    enabled = [metric for metric in selected if config.observations[metric].sources.live]
    numeric = [metric for metric in enabled if OBSERVATIONS[metric].kind == 'numeric']
    files = [metric for metric in enabled if OBSERVATIONS[metric].kind == 'file']
    result = {'status': 'complete', 'failed_metrics': [], 'downloaded_measurements': 0}
    if numeric:
        from clio.ingestion.collect_numeric import collect_numeric
        result = await collect_numeric(config, numeric, now=now)
    if files:
        from clio.ingestion.collect_files import collect_files
        report = await collect_files(config, files, mode='live', now=now)
        result['files'] = report['files']
        result['failed_metrics'] = sorted(set(result['failed_metrics']) | set(report['failed_metrics']))
        if report['status'] == 'partial':
            result['status'] = 'partial'
        elif report['status'] == 'busy' and result['status'] == 'complete':
            result['status'] = 'busy'
    return result
