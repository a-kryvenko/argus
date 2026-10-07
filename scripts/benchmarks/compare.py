"""Score exported point forecasts and both mandatory controls on identical pairs.

Predictions parquet: target, issue_time, valid_time, lead_hours, prediction.
This entrypoint scores points only; probability/quantile forecasts need their
corresponding proper scores as specified by the benchmark protocol.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import digest, score_pairs, write_json


KEYS = ['target', 'issue_time', 'valid_time', 'lead_hours']


def validate(frame, metadata, protocol, fold):
    required = ['model_sha256', 'train_start', 'fit_end', 'selection_end', 'calibration_end',
                'upstream_training', 'prior_test_exposure', 'target_definition']
    if any(key not in metadata for key in required):
        raise ValueError(f'Metadata requires {required}')
    if metadata['target_definition'] != protocol['id']:
        raise ValueError('Declare compatibility with benchmark target definitions')
    for key in ['train_start', 'fit_end', 'selection_end', 'calibration_end']:
        timestamp = pd.Timestamp(metadata[key])
        if timestamp.tzinfo is None:
            raise ValueError('Provenance timestamps must be timezone-aware')
    if pd.Timestamp(metadata['train_start']) != pd.Timestamp(protocol['train_start']):
        raise ValueError('Training start differs from controlled benchmark')
    cutoff = pd.Timestamp(protocol['folds'][fold]['train_end'])
    if any(pd.Timestamp(metadata[key]) > cutoff for key in ['fit_end', 'selection_end', 'calibration_end']):
        raise ValueError('Fitting, selection or calibration overlaps evaluation period')
    result = frame[[*KEYS, 'prediction']].copy()
    for key in ('issue_time', 'valid_time'):
        if any(pd.Timestamp(t).tzinfo is None for t in result[key]):
            raise ValueError('Forecast timestamps must be timezone-aware')
        result[key] = pd.to_datetime(result[key], utc=True)
        if result[key].isna().any():
            raise ValueError('Missing forecast timestamp')
    leads = pd.to_numeric(result.lead_hours, errors='raise')
    if not (leads.between(1, protocol['max_lead_hours']) & leads.eq(np.floor(leads))).all():
        raise ValueError('Invalid lead hours')
    result['lead_hours'] = leads.astype(int)
    if not (result.valid_time == result.issue_time + pd.to_timedelta(leads, unit='h')).all():
        raise ValueError('Inconsistent forecast times')
    if result.empty or result.duplicated(KEYS).any():
        raise ValueError('Empty or duplicate forecasts')
    # NaN predictions mean missing model coverage; infinities are invalid.
    result['prediction'] = pd.to_numeric(result.prediction, errors='raise')
    if np.isinf(result.prediction).any():
        raise ValueError('Infinite prediction')
    return result


def compare(benchmark, predictions, metadata_path, fold, output):
    manifest = json.loads((benchmark/'manifest.json').read_text())
    def verified(relative):
        path = benchmark/relative
        if digest(path) != manifest['artifacts'][relative]:
            raise ValueError(f'Changed benchmark artifact: {relative}')
        return path
    protocol = json.loads(verified('protocol.json').read_text())
    metadata = json.loads(metadata_path.read_text())
    forecasts = validate(pd.read_parquet(predictions), metadata, protocol, fold)
    climates = json.loads(verified('climatology.json').read_text())[fold]
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for (target, lead), group in forecasts.groupby(['target', 'lead_hours']):
        path = verified(f'pairs/{fold}/{target}/{lead:02d}.parquet')
        pairs = pd.read_parquet(path)
        keys = ['issue_time', 'valid_time', 'lead_hours']
        joined = pairs.merge(group[keys+['prediction']], on=keys, how='left', validate='one_to_one')
        matched_keys = group.merge(pairs[keys], on=keys, how='inner')
        if len(matched_keys) != len(group):
            raise ValueError('Forecast rows outside native target grid or evaluation split')
        joined.to_parquet(output/f'{target}-{lead:02d}.parquet', index=False)
        rows.extend(dict(target=target, lead_hours=int(lead), fold=fold,
                         n_model_finite=int(np.isfinite(joined.prediction).sum()), **row)
                    for row in score_pairs(joined, climates[target],
                                           protocol['thresholds'].get(target, []), model='prediction'))
    pd.DataFrame(rows).to_csv(output/'metrics.csv', index=False)
    write_json(output/'manifest.json', dict(benchmark_manifest_sha256=digest(benchmark/'manifest.json'),
        predictions_sha256=digest(predictions), metadata=metadata,
        metadata_sha256=digest(metadata_path), code_sha256=digest(Path(__file__)),
        scorer_sha256=digest(Path(__file__).with_name('benchmark.py')),
        scope='point forecasts only; omitted target/leads are not evaluated',
        artifacts={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('benchmark', 'predictions', 'metadata', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--fold', required=True)
    args = parser.parse_args()
    compare(args.benchmark, args.predictions, args.metadata, args.fold, args.output)
