"""UTC timestamp + internal variables or {var}_url; missing values are NaN.

Providers filter source flags before returning data. Ranges use [start, end).
"""
import numpy as np
import pandas as pd
import math

from clio.ingestion.utils.products import OBSERVATIONS

SAMPLE_COLUMNS = ['metric', 'observed_at', 'value', 'quality', 'spacecraft']


def sample(metric, observed_at, raw, *, flagged=False, spacecraft=None):
    """Preserve an explicit provider rejection at its observation time."""
    try:
        value = float(raw) if raw is not None and not isinstance(raw, bool) else math.nan
    except (TypeError, ValueError, OverflowError):
        value = math.nan
    quality = 'flagged' if flagged or (raw is not None and not OBSERVATIONS[metric].accepts(value)) else None
    if raw is None and quality is None:
        quality = 'missing'
    return dict(metric=metric, observed_at=observed_at, value=None if quality else value,
                quality=quality, spacecraft=spacecraft)


def samples_frame(records):
    frame = pd.DataFrame(records, columns=SAMPLE_COLUMNS)
    frame['observed_at'] = pd.to_datetime(frame.observed_at, utc=True)
    return frame.drop_duplicates(['metric', 'observed_at'], keep='last').sort_values(
        'observed_at', kind='stable').reset_index(drop=True)


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
