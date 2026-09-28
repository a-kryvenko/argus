"""Select observation kinds and combine independent numeric/file reports."""
from clio.ingestion.products import OBSERVATIONS


def partition(config, metrics):
    metrics = list(dict.fromkeys(metrics or config.observations))
    if not metrics or any(m not in config.observations for m in metrics):
        raise ValueError('Select observations configured in clio.observations')
    return ([m for m in metrics if OBSERVATIONS[m].kind == 'numeric'],
            [m for m in metrics if OBSERVATIONS[m].kind == 'file'])


def merge_files(result, files):
    return {**result, 'files': files['files'],
            'status': 'partial' if 'partial' in (result['status'], files['status']) else 'complete',
            'failed_metrics': sorted(set(result['failed_metrics']) | set(files['failed_metrics']))}


def available_files(result):
    return sum(f['received'] + f['restored'] + f['retained'] for f in result.get('files', {}).values())
