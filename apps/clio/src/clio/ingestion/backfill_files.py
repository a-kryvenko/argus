"""Fill configured historical file observations."""
from clio.ingestion.utils.products import OBSERVATIONS


async def backfill_files(config, metrics, *, now):
    selected = list(dict.fromkeys(metrics))
    if any(metric not in config.observations or OBSERVATIONS[metric].kind != 'file' for metric in selected):
        raise ValueError('Select configured file observations')
    from clio.ingestion.utils.file_collection import collect_observation
    result = {'status': 'complete', 'failed_metrics': [], 'files': {}}
    for metric in selected:
        report = await collect_observation(config, metric, mode='backfill', now=now)
        result['files'][metric] = report
        if report['status'] == 'partial':
            result['status'] = 'partial'
            result['failed_metrics'].append(metric)
    return result
