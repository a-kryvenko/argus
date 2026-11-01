"""Published SET solar means, dated by the end of their 81-day window."""
from io import StringIO

import numpy as np
import pandas as pd
import requests

SOLFSMY_URL = 'https://sol.spacenvironment.net/JB2008/indices/SOLFSMY.TXT'
MEAN_COLUMNS = {'f10_7_81mean': 4, 's10_81mean': 6, 'm10_81mean': 8, 'y10_81mean': 10}


def parse_solfsmy_means(text):
    raw = pd.read_csv(StringIO(text), sep=r'\s+', comment='#', header=None)
    if raw.shape[1] not in (11, 12):
        raise ValueError('Unexpected SOLFSMY format (expected 11 or 12 columns)')
    years = pd.to_numeric(raw[0], errors='raise').astype(int)
    days = pd.to_numeric(raw[1], errors='raise').astype(int)
    dates = pd.to_datetime(years.astype(str) + days.astype(str).str.zfill(3), format='%Y%j', utc=True)
    if not dates.dt.year.eq(years).all() or dates.duplicated().any():
        raise ValueError('Invalid or duplicate SOLFSMY dates')
    # A centered mean at D covers D-40..D+40. Preserve that window end;
    # do not label the held value as an observation for the current day.
    window_end = dates + pd.Timedelta(days=40, hours=12)
    frames = []
    for metric, column in MEAN_COLUMNS.items():
        values = pd.to_numeric(raw[column], errors='coerce')
        valid = np.isfinite(values) & values.gt(0)
        frames.append(pd.DataFrame(dict(metric=metric, value=values[valid], observed_at=window_end[valid])))
    return pd.concat(frames, ignore_index=True).sort_values(['observed_at', 'metric']).reset_index(drop=True)


def fetch_solfsmy_means():
    response = requests.get(SOLFSMY_URL, timeout=120)
    response.raise_for_status()
    return parse_solfsmy_means(response.text)
