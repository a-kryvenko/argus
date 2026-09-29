"""Dispatch file observations to their owning domains."""


async def collect_file_observations(config, metrics, **kwargs):
    from clio.domains.aia.collection import collect_file_observations as collect_aia
    from clio.domains.gong import collect as collect_gong
    result = {'status': 'complete', 'failed_metrics': [], 'files': {}}
    for metric in metrics:
        if metric == 'aia193':
            report = await collect_aia(config, [metric], **kwargs)
        elif metric == 'gong':
            report = await collect_gong(config, **kwargs)
        else:
            raise ValueError(f'Unknown file observation: {metric}')
        result['files'].update(report['files'])
        result['failed_metrics'].extend(report['failed_metrics'])
        if report['status'] == 'partial':
            result['status'] = 'partial'
    return result
