"""Shared physical-unit OMNI missing-value contract for ingestion and training."""
import pandas as pd

# Hourly OMNI2 fill values; Kp in persisted CSVs is already divided by ten.
# https://omniweb.gsfc.nasa.gov/html/ow_data.html
OMNI_FILL_VALUES = {
    "bx": 999.9, "by": 999.9, "bz": 999.9, "t": 9999999,
    "n": 999.9, "v": 9999, "kp": 9.9, "dst": 99999,
    "ap": 999, "f10_7": 999.9,
}


def clean_omni_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean physical-unit OMNI columns without filling missing observations."""
    frame = frame.copy()
    for column, fill in OMNI_FILL_VALUES.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = values.mask(values.abs().eq(fill))
    return frame


OMNI_WORDS = {'bx': 13, 'by': 16, 'bz': 17, 't': 23, 'n': 24, 'v': 25,
         'kp': 39, 'dst': 41, 'ap': 50, 'f10_7': 51}


def parse_annual(text, year):
    import io
    table = pd.read_csv(io.StringIO(text), sep=r'\s+', header=None)
    if table.shape[1] < max(OMNI_WORDS.values()) or table.empty:
        raise ValueError('Unexpected annual OMNI schema')
    if not table[0].eq(year).all() or not table[2].between(0, 23).all():
        raise ValueError('Invalid year/hour in annual OMNI file')
    times = pd.to_datetime(table[0].astype(str) + table[1].astype(str).str.zfill(3),
                           format='%Y%j', utc=True) + pd.to_timedelta(table[2], unit='h')
    if times.duplicated().any() or not times.dt.year.eq(year).all():
        raise ValueError('Duplicate or invalid observation timestamps')
    frame = pd.DataFrame({key: pd.to_numeric(table[word-1], errors='raise') for key, word in OMNI_WORDS.items()})
    frame['kp'] = frame['kp'] / 10
    frame = clean_omni_values(frame)
    frame.insert(0, 'issue_time', times)
    return frame.sort_values('issue_time').reset_index(drop=True)

