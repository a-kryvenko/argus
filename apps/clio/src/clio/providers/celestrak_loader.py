"""Observed F10.7 and published trailing 81-day means from CelesTrak.

Format: https://celestrak.org/SpaceData/SpaceWx-format.php
"""
from io import StringIO

import numpy as np
import pandas as pd
import requests

CELESTRAK_RECENT_URL = 'https://celestrak.org/SpaceData/SW-Last5Years.csv'
CELESTRAK_HISTORY_URL = 'https://celestrak.org/SpaceData/SW-All.csv'
METRICS = {'f10_7': 'F10.7_OBS', 'f10_7_81mean': 'F10.7_OBS_LAST81'}


def parse_celestrak_f107(text: str) -> pd.DataFrame:
    raw = pd.read_csv(StringIO(text))
    required = {'DATE', 'F10.7_DATA_TYPE', *METRICS.values()}
    if not required.issubset(raw.columns):
        raise ValueError('Unexpected CelesTrak CSV format; missing: ' + ', '.join(sorted(required - set(raw.columns))))
    # Forecasts and interpolated daily flux are not observed measurements.
    raw = raw.loc[raw['F10.7_DATA_TYPE'].eq('OBS')].copy()
    dates = pd.to_datetime(raw.DATE, format='%Y-%m-%d', utc=True)
    if dates.duplicated().any():
        raise ValueError('Duplicate observation dates in CelesTrak CSV')
    # Ottawa used 17 UT; Penticton has used 20 UT since June 1991.
    hours = np.where(dates < pd.Timestamp('1991-06-01', tz='UTC'), 17, 20)
    observed_at = dates + pd.to_timedelta(hours, unit='h')
    frames = []
    for metric, column in METRICS.items():
        values = pd.to_numeric(raw[column], errors='coerce')
        valid = np.isfinite(values) & values.gt(0)
        frames.append(pd.DataFrame({'metric': metric, 'value': values[valid],
                                    'observed_at': observed_at[valid]}))
    return pd.concat(frames, ignore_index=True).sort_values(['observed_at', 'metric']).reset_index(drop=True)


def fetch_celestrak_f107(*, url=CELESTRAK_RECENT_URL) -> pd.DataFrame:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return parse_celestrak_f107(response.text)
