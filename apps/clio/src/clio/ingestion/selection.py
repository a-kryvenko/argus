"""Select observation kinds and combine independent numeric/file reports."""
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


def available_files(result):
    images = sum(result.get('sdo', {}).get(key, 0) for key in ('saved', 'restored', 'existing'))
    return images + sum(f['received'] + f['restored'] + f['retained']
                        for f in result.get('files', {}).values())


async def merge_sdo(result, config, args, *, mode):
    if not config.sdo_images.enabled or (args.metrics is not None and 'sdo' not in args.metrics):
        return result
    import asyncio
    from clio.commands.sdo_images import cycle
    from common.sdo_images import archive_root
    report = await asyncio.to_thread(
        cycle, archive_root(), mode,
        batch_hours=config.sdo_images.warmup_hours_per_batch,
        full=not getattr(args, 'scheduled', False),
        start=getattr(args, 'start', None), end=getattr(args, 'end', None))
    result['sdo'] = report
    if report.get('failed'):
        result['status'] = 'partial'
        result['failed_metrics'] = sorted(set(result['failed_metrics']) | {'sdo'})
    return result
