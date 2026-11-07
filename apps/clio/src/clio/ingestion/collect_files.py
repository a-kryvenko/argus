"""Collect live file observations and source catalogs."""
import logging

from clio.ingestion.utils.products import OBSERVATIONS

logger = logging.getLogger(__name__)


async def collect_files(config, metrics=None, *, now):
    selected = ([metric for metric in config.observations if OBSERVATIONS[metric].kind == 'file'
                and config.observations[metric].sources.live] + ['donki']) if metrics is None else list(dict.fromkeys(metrics))
    if any(metric != 'donki' and
           (metric not in config.observations or OBSERVATIONS[metric].kind != 'file') for metric in selected):
        raise ValueError('Select configured file observations')
    result = {'status': 'complete', 'failed_metrics': []}
    for metric in selected:
        if metric == 'donki':
            continue
        from clio.ingestion.utils.file_collection import collect_observation
        report = await collect_observation(config, metric, mode='live', now=now)
        result.setdefault('files', {})[metric] = report
        if report['status'] == 'partial':
            result['status'] = 'partial'
            result['failed_metrics'].append(metric)
    if 'donki' in selected:
        if not config.solar_wind_events.enabled:
            result['donki'] = {'status': 'disabled'}
        else:
            from clio.ingestion.extras import donki
            try:
                await donki.collect(config.solar_wind_events, now=now)
            except RuntimeError as exc:
                logger.warning('DONKI collection failed: %s', exc)
                result['donki'] = {'status': 'failed'}
                result['status'] = 'partial'
                result['failed_metrics'].append('donki')
            else:
                result['donki'] = {'status': 'complete'}
    result['failed_metrics'] = sorted(set(result['failed_metrics']))
    return result
