"""Download/cache annual hourly OMNI ASCII files and unfilled parquet observations.

Usage: python -m clio.dataloaders.omni_archive --data-root data
Schema: https://omniweb.gsfc.nasa.gov/html/ow_data.html (one-based word numbers).
"""
import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from common.data.omni import clean_omni_values

BASE_URL = 'https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni'
WORDS = {'bx': 13, 'by': 16, 'bz': 17, 't': 23, 'n': 24, 'v': 25,
         'kp': 39, 'dst': 41, 'ap': 50, 'f10_7': 51}


def parse_annual(text, year):
    table = pd.read_csv(io.StringIO(text), sep=r'\s+', header=None)
    if table.shape[1] < max(WORDS.values()) or table.empty:
        raise ValueError('Unexpected annual OMNI schema')
    if not table[0].eq(year).all() or not table[2].between(0, 23).all():
        raise ValueError('Invalid year/hour in annual OMNI file')
    times = pd.to_datetime(table[0].astype(str) + table[1].astype(str).str.zfill(3),
                           format='%Y%j', utc=True) + pd.to_timedelta(table[2], unit='h')
    if times.duplicated().any() or not times.dt.year.eq(year).all():
        raise ValueError('Duplicate or invalid observation timestamps')
    frame = pd.DataFrame({key: pd.to_numeric(table[word-1], errors='raise') for key, word in WORDS.items()})
    frame['kp'] = frame['kp'] / 10
    frame = clean_omni_values(frame)
    frame.insert(0, 'issue_time', times)
    return frame.sort_values('issue_time').reset_index(drop=True)


def download_archive(data_root, start=1963, end=None):
    end = end or datetime.now(timezone.utc).year
    if not 1963 <= start <= end <= datetime.now(timezone.utc).year:
        raise ValueError('Invalid year range')
    raw_dir = Path(data_root) / 'raw/omni_hourly'
    clean_dir = Path(data_root) / 'clean/omni_hourly'
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
                   status_forcelist=[429, 500, 502, 503, 504])))
    manifest = []
    # Acquire the explicitly requested missing decade first.
    years = sorted(range(start, end+1), key=lambda y: (not 2000 <= y <= 2009, y))
    for year in years:
        url = f'{BASE_URL}/omni2_{year}.dat'
        raw_path = raw_dir / f'omni2_{year}.dat'
        if raw_path.exists():
            content = raw_path.read_bytes()
        else:
            response = session.get(url, timeout=(20, 120))
            response.raise_for_status()
            content = response.content
        frame = parse_annual(content.decode('ascii'), year)
        if not raw_path.exists():
            tmp = raw_path.with_suffix('.part'); tmp.write_bytes(content); tmp.replace(raw_path)
        destination = clean_dir / f'omni_{year}.parquet'
        tmp = destination.with_suffix('.part'); frame.to_parquet(tmp, index=False); tmp.replace(destination)
        manifest.append(dict(year=year, source=url, sha256=hashlib.sha256(content).hexdigest(),
                             rows=len(frame), observed_v=int(frame.v.notna().sum()),
                             fetched_or_verified_utc=datetime.now(timezone.utc).isoformat()))
        (raw_dir / 'manifest.json').write_text(json.dumps(sorted(manifest, key=lambda r:r['year']), indent=2))
        print(f'{year}: {len(frame)} rows, {frame.v.notna().mean():.1%} observed v', flush=True)
    if start <= 2000 and end >= 2009:
        decade = pd.concat([pd.read_parquet(clean_dir/f'omni_{y}.parquet') for y in range(2000,2010)])
        # Compatible with the existing normalize/process/build notebook pipeline.
        for folder in ['raw', 'clean']:
            decade.to_csv(Path(data_root)/folder/'omni_2000_2009.csv', index=False)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--start', type=int, default=1963)
    parser.add_argument('--end', type=int)
    args = parser.parse_args()
    download_archive(args.data_root, args.start, args.end)
