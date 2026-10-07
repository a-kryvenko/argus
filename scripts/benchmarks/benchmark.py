"""Offline observation preparation and frozen persistence/climatology evaluation.

Run: python -m scripts.benchmarks.benchmark --output data/metrics/benchmarks/RUN
No training code, model weights, source data or earlier reports are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.data.omni import clean_omni_values

HERE = Path(__file__).resolve().parent
SPECS = {
    'v': ('km/s', 'hourly'), 'n': ('cm^-3', 'hourly'), 't': ('K', 'hourly'),
    'bt': ('nT', 'hourly'), 'bs': ('nT', 'hourly'), 'dst': ('nT', 'hourly'),
    'kp': ('index', 'three_hourly'), 'ap': ('nT', 'three_hourly'),
    'f107': ('sfu', 'daily'), 's10': ('sfu-equivalent', 'daily'),
    'm10': ('sfu-equivalent', 'daily'), 'y10': ('sfu-equivalent', 'daily'),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def observation_rows(times, values, target, source, latency):
    unit, cadence = SPECS[target]
    hours = {'hourly': 1, 'three_hourly': 3, 'daily': 24}[cadence]
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    end = times + pd.Timedelta(hours=hours)
    return pd.DataFrame(dict(target=target, valid_time=times, interval_end=end,
        available_at=end + pd.Timedelta(hours=latency),
        value=np.asarray(values, dtype=float), unit=unit, cadence=cadence, source=source,
        availability_basis='assumed_interval_end_plus_latency'))


def load_observations(root, protocol):
    """Keep native cadence and missing labels; never use interpolated training data."""
    start = pd.Timestamp(protocol['train_start'])
    end = pd.Timestamp(protocol['as_of'])
    parts, sources = [], []
    for year in range(start.year, end.year + 1):
        path = root / f'clean/omni_hourly/omni_{year}.parquet'
        if not path.exists():
            raise FileNotFoundError(path)
        sources.append(dict(path=str(path), sha256=digest(path)))
        frame = clean_omni_values(pd.read_parquet(path)).replace([np.inf, -np.inf], np.nan)
        times = pd.to_datetime(frame.issue_time, utc=True)
        if times.duplicated().any():
            raise ValueError(f'Duplicate OMNI times: {path}')
        # Explicit definition: magnitude of the hourly mean vector, not mean |B|.
        frame['bt'] = np.sqrt(frame.bx**2 + frame.by**2 + frame.bz**2)
        frame['bs'] = (-frame.bz).clip(lower=0)
        for target in ('v', 'n', 't', 'bt', 'bs', 'dst'):
            parts.append(observation_rows(times, frame[target], target, 'omni-hourly',
                                          protocol['latency_hours']['omni']))
        for target in ('kp', 'ap'):
            native = pd.DataFrame({'time': times, 'value': frame[target]}).assign(
                block=times.dt.floor('3h'))
            counts = native.groupby('block').value.nunique()
            if counts.gt(1).any():
                raise ValueError(f'Conflicting repeated {target} values: {path}')
            native = native.groupby('block').value.first()
            parts.append(observation_rows(native.index, native, target, 'omni-three-hourly',
                                          protocol['latency_hours']['omni']))
    for path in sorted((root / 'raw/spacewx').glob('solfsmy_*.parquet')):
        sources.append(dict(path=str(path), sha256=digest(path)))
        frame = pd.read_parquet(path)
        # Daily target date; do not replicate a daily observation into hourly labels.
        times = pd.to_datetime(frame.timestamp, utc=True).dt.floor('D')
        for target, column in [('f107', 'f10'), ('s10', 's10'), ('m10', 'm10'), ('y10', 'y10')]:
            values = pd.to_numeric(frame[column], errors='raise')
            values = values.where(np.isfinite(values) & values.gt(0) & values.ne(999.9))
            parts.append(observation_rows(times, values, target, 'solfsmy-daily',
                                          protocol['latency_hours']['solfsmy']))
    result = pd.concat(parts, ignore_index=True)
    result = result[(result.valid_time >= start) & (result.interval_end <= end)]
    if result.duplicated(['target', 'valid_time']).any():
        raise ValueError('Overlapping observation sources; resolve explicitly')
    result = result.sort_values(['target', 'valid_time']).reset_index(drop=True)
    return result, sources


def fit_climatology(observations, start, end, thresholds):
    selected = observations[(observations.valid_time >= pd.Timestamp(start)) &
                            (observations.interval_end <= pd.Timestamp(end)) &
                            (observations.available_at <= pd.Timestamp(end))]
    values = selected.value.to_numpy(float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return dict(n=len(values), mean=float(values.mean()),
        quantiles=[float(x) for x in np.quantile(values, [.1, .5, .9])],
        probabilities={str(t): float((values >= t).mean()) for t in thresholds})


def target_pairs(observations, fold, protocol, lead):
    """One issue per native target and exact lead; all intervals stay inside split."""
    start = pd.Timestamp(fold['test_start'])
    end = min(pd.Timestamp(fold['test_end']), pd.Timestamp(protocol['as_of']))
    cadence = observations.cadence.iloc[0]
    frequency = {'hourly': 'h', 'three_hourly': '3h', 'daily': 'D'}[cadence]
    times = pd.date_range(start, end, freq=frequency, inclusive='left')
    pairs = pd.DataFrame({'valid_time': times})
    pairs['issue_time'] = pairs.valid_time - pd.Timedelta(hours=lead)
    hours = {'hourly': 1, 'three_hourly': 3, 'daily': 24}[cadence]
    pairs = pairs[(pairs.issue_time >= start) &
                  (pairs.valid_time + pd.Timedelta(hours=hours) <= end)].copy()
    stride = protocol['issue_stride_hours']
    pairs = pairs[((pairs.issue_time.astype('int64') // 3_600_000_000_000) % stride) == 0]
    pairs = pairs.merge(observations[['valid_time', 'value']], on='valid_time', how='left', validate='one_to_one')
    history = observations[np.isfinite(observations.value)].sort_values('available_at')
    history = history[['available_at', 'valid_time', 'value']].rename(
        columns={'valid_time': 'history_valid_time', 'value': 'persistence'})
    pairs = pd.merge_asof(pairs.sort_values('issue_time'), history,
                          left_on='issue_time', right_on='available_at', direction='backward')
    age = (pairs.issue_time - pairs.history_valid_time) / pd.Timedelta(hours=1)
    pairs.loc[age > protocol['max_persistence_age_hours'][cadence], 'persistence'] = np.nan
    pairs['lead_hours'] = lead
    return pairs


def score_pairs(pairs, climate, thresholds, model=None):
    """Both baselines and optional model are scored on exactly the same finite rows."""
    mask = np.isfinite(pairs.value) & np.isfinite(pairs.persistence)
    if model is not None:
        mask &= np.isfinite(pairs[model])
    n = int(mask.sum()) if climate is not None else 0
    counts = dict(n_expected=len(pairs), n_observed=int(np.isfinite(pairs.value).sum()), n=n,
                  n_unique_valid_times=int(pairs.loc[mask, 'valid_time'].nunique()) if n else 0)
    if not n:
        return [dict(baseline=b, status='no_matched_data', **counts, mae=None, rmse=None)
                for b in ['persistence', 'climatology'] + ([model] if model else [])]
    y = pairs.loc[mask, 'value'].to_numpy()
    predictions = {'persistence': pairs.loc[mask, 'persistence'].to_numpy(),
                   'climatology': np.full(n, climate['mean'])}
    if model:
        predictions[model] = pairs.loc[mask, model].to_numpy()
    rows = []
    for name, prediction in predictions.items():
        error = prediction - y
        row = dict(baseline=name, status='ok', **counts, mae=float(np.abs(error).mean()),
                   rmse=float(np.sqrt(np.square(error).mean())), bias=float(error.mean()))
        if name != model:
            quantiles = np.tile(climate['quantiles'], (n, 1)) if name == 'climatology' else np.repeat(prediction[:, None], 3, axis=1)
            residual = y[:, None] - quantiles
            for i, q in enumerate((.1, .5, .9)):
                row[f'pinball_q{int(q*100)}'] = float(np.maximum(q*residual[:, i], (q-1)*residual[:, i]).mean())
            row['coverage80'] = float(((y >= quantiles[:, 0]) & (y <= quantiles[:, 2])).mean())
            row['width80'] = float((quantiles[:, 2]-quantiles[:, 0]).mean())
            for threshold in thresholds:
                probability = climate['probabilities'][str(threshold)] if name == 'climatology' else (prediction >= threshold).astype(float)
                row[f'brier_ge_{threshold}'] = float(np.square(probability - (y >= threshold)).mean())
        rows.append(row)
    if model:
        for metric in ('mae', 'rmse'):
            for baseline in rows[:2]:
                denom = baseline[metric]
                rows[-1][f'{metric}_skill_vs_{baseline["baseline"]}'] = 1 - rows[-1][metric]/denom if denom else None
    return rows


def coverage_report(observations, protocol):
    rows = []
    for target in SPECS:
        group = observations[observations.target.eq(target)]
        for year in range(pd.Timestamp(protocol['train_start']).year, pd.Timestamp(protocol['as_of']).year+1):
            part = group[group.valid_time.dt.year.eq(year)]
            valid = part[np.isfinite(part.value)]
            rows.append(dict(target=target, year=year, n_rows=len(part), n_observed=len(valid),
                first=str(valid.valid_time.min()) if len(valid) else None,
                last=str(valid.valid_time.max()) if len(valid) else None,
                status='available' if len(valid) else 'missing'))
    return pd.DataFrame(rows)


def pilot_coverage(observations, path, protocol):
    """Audit every pilot split without changing its training or evaluating weights."""
    plan = json.loads(path.read_text())
    rows = []
    for split in ('train', 'validation', 'test'):
        selected = [r for r in plan['rows'] if r['split'] == split]
        issues = pd.DatetimeIndex([(pd.Timestamp(r['observed_at']) + pd.Timedelta(hours=plan['latency_hours'])).ceil('6h') for r in selected])
        ends = pd.DatetimeIndex([r['split_end'] for r in selected])
        for target in SPECS:
            truth = observations[observations.target.eq(target)].set_index('valid_time').value
            cadence = SPECS[target][1]
            duration = {'hourly': 1, 'three_hourly': 3, 'daily': 24}[cadence]
            for lead in range(1, protocol['max_lead_hours']+1):
                times = issues + pd.Timedelta(hours=lead)
                native = (times.hour % duration == 0)
                within = times + pd.Timedelta(hours=duration) <= ends
                available = np.isfinite(truth.reindex(times).to_numpy()) & native & within
                rows.append(dict(split=split, target=target, lead_hours=lead, n_issues=len(issues),
                    n_native_in_split=int((native & within).sum()), n_observed=int(available.sum())))
    return pd.DataFrame(rows)


def run(root, output, protocol_path, pilot_plan=None):
    protocol = json.loads(protocol_path.read_text())
    # Exclusive directory creation protects previously frozen reports, including failed runs.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'status.json', {'status': 'running'})
    write_json(output/'protocol.json', protocol)
    observations, sources = load_observations(root, protocol)
    observations.to_parquet(output/'observations.parquet', index=False)
    coverage_report(observations, protocol).to_csv(output/'coverage.csv', index=False)
    if pilot_plan:
        pilot_coverage(observations, pilot_plan, protocol).to_csv(output/'pilot_coverage.csv', index=False)
        sources.append(dict(path=str(pilot_plan), sha256=digest(pilot_plan)))
    rows, fitted = [], {}
    for fold_name, fold in protocol['folds'].items():
        fitted[fold_name] = {}
        for target in SPECS:
            group = observations[observations.target.eq(target)]
            thresholds = protocol['thresholds'].get(target, [])
            climate = fit_climatology(group, protocol['train_start'], fold['train_end'], thresholds)
            fitted[fold_name][target] = climate
            if group.empty:
                group = observation_rows([], [], target, 'missing', 0)
                # Preserve cadence for constructing a missing-data grid.
                rows.extend(dict(fold=fold_name, target=target, baseline=b, lead_hours=h,
                    status='no_observation_source', n=0) for b in protocol['required_baselines']
                    for h in range(1, protocol['max_lead_hours']+1))
                continue
            destination = output/'pairs'/fold_name/target
            destination.mkdir(parents=True)
            for lead in range(1, protocol['max_lead_hours']+1):
                pairs = target_pairs(group, fold, protocol, lead)
                pairs.to_parquet(destination/f'{lead:02d}.parquet', index=False)
                rows.extend(dict(fold=fold_name, target=target, unit=SPECS[target][0], lead_hours=lead, **r)
                            for r in score_pairs(pairs, climate, thresholds))
            print(f'{fold_name}: {target}', flush=True)
    write_json(output/'climatology.json', fitted)
    pd.DataFrame(rows).to_csv(output/'metrics.csv', index=False)
    write_json(output/'unsupported.json', {target: dict(status='missing_evidence', reason=reason,
        required_baselines=protocol['required_baselines']) for target, reason in protocol['unsupported_targets'].items()})
    artifacts = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'status.json'}
    write_json(output/'manifest.json', dict(protocol_sha256=digest(protocol_path),
        code_sha256=digest(Path(__file__)), sources=sources, artifacts=artifacts))
    write_json(output/'status.json', {'status': 'complete', 'model_evaluation': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=HERE/'protocol.json')
    parser.add_argument('--pilot-plan', type=Path)
    args = parser.parse_args()
    run(args.data_root, args.output, args.protocol, args.pilot_plan)
