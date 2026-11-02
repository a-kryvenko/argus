"""UTC timestamp + internal variables or {var}_url; missing values are NaN.

Providers filter source flags before returning data. Ranges use [start, end).
"""
import numpy as np
import pandas as pd


def dataset(frame, variables):
    frame = frame.reindex(columns=['timestamp', *variables]).copy()
    frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True)
    frame = frame.drop_duplicates('timestamp', keep='last')
    for name in variables:
        if name.endswith('_url'):
            continue
        values = pd.to_numeric(frame[name], errors='coerce')
        valid = np.isfinite(values)
        if name not in ('bx', 'by', 'bz', 'dst'):
            valid &= values.ge(0)
        if name == 'kp':
            valid &= values.le(9)
        frame[name] = values.where(valid)
    return (frame.dropna(subset=['timestamp']).dropna(subset=list(variables), how='all')
            .sort_values('timestamp').reset_index(drop=True))


def from_measurements(frame, variables):
    wide = frame.pivot(index='observed_at', columns='metric', values='value').reset_index()
    wide.columns.name = None
    return dataset(wide.rename(columns={'observed_at': 'timestamp'}), variables)
