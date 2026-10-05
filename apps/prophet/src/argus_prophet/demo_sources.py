"""Prepare causal replay snapshots for the demo catalogue from NASA OMNI archives.

Run from the repository root with the Prophet Python environment. No downloads,
training, target interpolation, operational publication or database writes are performed.
"""
import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from argus_prophet.demo import Manifest, issue_times
from common.data.omni import parse_annual


def prepare(manifest_path, archives, destination):
    manifest = Manifest.model_validate_json(manifest_path.read_text())
    frames, sources = [], []
    for archive in archives:
        content = archive.read_bytes()
        if archive.suffix == '.parquet':
            frame = pd.read_parquet(archive)
        else:
            text = content.decode('ascii')
            frame = parse_annual(text, int(text.split()[0]))
        frames.append(frame)
        sources.append({'file': archive.name, 'sha256': hashlib.sha256(content).hexdigest()})
    history = pd.concat(frames).sort_values('issue_time')
    history['issue_time'] = pd.to_datetime(history.issue_time, utc=True)
    if history.issue_time.duplicated().any():
        raise ValueError('Input archives overlap')
    import numpy as np
    history['bt'] = np.sqrt(history.bx ** 2 + history.by ** 2 + history.bz ** 2)
    history['bs'] = (-history.bz).clip(lower=0)
    destination.mkdir(parents=True, exist_ok=True)
    snapshots = destination / 'snapshots'
    snapshots.mkdir(exist_ok=True)
    read_at = datetime.now(UTC).isoformat()
    for index, issue in enumerate(issue_times(manifest.event)):
        frame = history.loc[(history.issue_time < issue) & (history.issue_time >= issue - timedelta(days=88))]
        payload = {'as_of': issue.isoformat(), 'read_at': read_at, 'observations': {'points': []}}
        for variable, key in [('v', 'speed_observations'), ('n', 'density_observations')]:
            rows = frame[['issue_time', variable]].dropna()
            payload[key] = [{'issue_time': row.issue_time.isoformat(), variable: float(getattr(row, variable))}
                            for row in rows.itertuples()]
        if set(manifest.products) - {'solar-wind-speed', 'solar-wind-density'}:
            payload['observations']['points'] = normalized_points(frame, issue)
        if 'hmf_total_threshold' in manifest.models:
            imf = frame.loc[frame.issue_time >= issue - timedelta(hours=168)]
            payload['solar_wind_hourly'] = {'schema': 'omni-hourly-v1', 'rows': [
                {'time': row.issue_time.isoformat(), **{key: None if pd.isna(getattr(row, key)) else float(getattr(row, key))
                  for key in ('bx', 'by', 'bz', 'v', 'n', 't')}} for row in imf.itertuples()]}
        (snapshots / f'{index:03d}.json').write_text(json.dumps(payload, allow_nan=False))
    start, finish = issue_times(manifest.event)[0] - timedelta(hours=24), issue_times(manifest.event)[-1] + timedelta(hours=96)
    selected = history.loc[history.issue_time.between(start, finish)]
    if selected.empty or selected.issue_time.min() > start or selected.issue_time.max() < finish:
        raise ValueError('Archives do not yet cover the complete event and forecast outcome window')
    observations = [{'time': row.issue_time.isoformat(), 'values': {
        variable: None if pd.isna(getattr(row, variable)) else float(getattr(row, variable)) for variable in ('v', 'n', 'kp', 'ap', 'dst', 'bt', 'bs', 'f10_7')}}
        for row in selected.itertuples()]
    (destination / 'observations.json').write_text(json.dumps(observations, allow_nan=False))
    (destination / 'sources.json').write_text(json.dumps(sources, indent=2))
    print(f'Prepared 97 input snapshots and {len(observations)} observation hours in {destination}')


def normalized_points(frame, issue):
    """Match the scalar input contract; normalize only data known by this issue.

    Kp/Ap refer to completed three-hour bins. Daily F10.7 is admitted only after
    the UTC day closes. Interpolation/means use this truncated history only;
    targets and speed/density raw histories are never filled.
    """
    keys = ['bx', 'by', 'bz', 'v', 'n', 't', 'kp', 'ap', 'dst', 'f10_7']
    wide = frame.loc[frame.issue_time < issue].set_index('issue_time')[keys].copy().asfreq('h')
    wide.loc[wide.index.floor('3h') + pd.Timedelta(hours=3) > issue, ['kp', 'ap']] = float('nan')
    wide.loc[wide.index.floor('D') + pd.Timedelta(days=1) > issue, 'f10_7'] = float('nan')
    # Forward filling at the trailing edge uses the last available index value.
    wide = wide.interpolate(method='time', limit_area='inside').ffill()
    wide = wide.fillna(wide.mean()).dropna()
    if len(wide) < 168 or wide.index[-1] != issue - timedelta(hours=1):
        raise ValueError('Insufficient complete historical input hours for geomagnetic/radiation models')
    return [{'issue_time': time.isoformat(), **{
        key: int(row[key]) if key in ('kp', 'ap', 'dst', 'f10_7') else float(row[key]) for key in keys}}
        for time, row in wide.iterrows()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--archive', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.archive, args.output)
