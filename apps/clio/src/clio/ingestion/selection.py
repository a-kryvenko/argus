"""Select observation kinds and combine independent numeric/file reports."""
from datetime import timedelta

from clio.ingestion.products import OBSERVATIONS


def partition(config, metrics):
    metrics = list(dict.fromkeys(config.observations if metrics is None else metrics))
    metrics = [m for m in metrics if m != 'sdo']
    if any(m not in config.observations for m in metrics):
        raise ValueError('Select observations configured in clio.observations')
    return ([m for m in metrics if OBSERVATIONS[m].kind == 'numeric'],
            [m for m in metrics if OBSERVATIONS[m].kind == 'file'])


def merge_files(result, files):
    return {**result, 'files': files['files'],
            'status': 'partial' if 'partial' in (result['status'], files['status']) else 'complete',
            'failed_metrics': sorted(set(result['failed_metrics']) | set(files['failed_metrics']))}


async def merge_sdo(result, config, *, now, mode, metrics=None):
    if not config.sdo_images.enabled or (metrics is not None and 'sdo' not in metrics):
        return result
    import asyncio
    from clio.ingestion.sdo_images import cycle
    from clio.observations.sdo_archive import archive_root
    report = await asyncio.to_thread(
        cycle, archive_root(), mode,
        full=True, clock=lambda: now,
        start=now - timedelta(days=config.sdo_images.backfill_days) if mode == 'warmup' else None,
        end=now if mode == 'warmup' else None)
    result['sdo'] = report
    if report.get('failed'):
        result['status'] = 'partial'
        result['failed_metrics'] = sorted(set(result['failed_metrics']) | {'sdo'})
    return result
