"""Fetch configured numeric/file observations without running normalization."""
import logging

from clio.ingestion.selection import partition, merge_files, merge_sdo

logger = logging.getLogger(__name__)


async def collect(config, metrics=None, *, now):
    browse = config.sdo_browse_enabled and (metrics is None or 'sdo-browse' in metrics)
    if metrics is not None:
        metrics = [metric for metric in metrics if metric != 'sdo-browse']
    catalogs = ('donki', 'enlil')
    selected = catalogs if metrics is None else [kind for kind in catalogs if kind in metrics]
    metrics = None if metrics is None else [metric for metric in metrics if metric not in catalogs]
    numeric, files = partition(config, metrics)
    result = {'status': 'complete', 'failed_metrics': [], 'downloaded_measurements': 0}
    if numeric:
        from clio.ingestion.numeric_live import collect_live
        result = await collect_live(config, numeric, now=now)
    if files:
        from clio.ingestion.files import collect_file_observations
        result = merge_files(result, await collect_file_observations(config, files, mode='live', now=now))
    result = await merge_sdo(result, config, metrics=metrics, now=now, mode='live')
    if browse:
        import asyncio
        from clio.observations.sdo_browse import collect as collect_browse
        report = await asyncio.to_thread(collect_browse, now)
        result['sdo_browse'] = report
        if report['failed']:
            result['status'] = 'partial'
            result['failed_metrics'] = sorted(set(result['failed_metrics']) | {'sdo-browse'})
    for kind in selected:
        if not config.solar_wind_events.enabled:
            result[kind] = {'status': 'disabled'}
            continue
        from clio.ingestion import donki, enlil
        try:
            if kind == 'donki':
                await donki.collect(config.solar_wind_events, now=now)
            else:
                await enlil.collect(now=now)
        except RuntimeError as exc:
            logger.warning('%s collection failed: %s', kind, exc)
            result[kind] = {'status': 'failed'}
            result['status'] = 'partial'
            result['failed_metrics'] = sorted(set(result['failed_metrics']) | {kind})
        else:
            result[kind] = {'status': 'complete'}
    return result
