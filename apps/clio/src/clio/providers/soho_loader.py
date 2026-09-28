"""SOHO/CELIAS PM five-minute proton moments, at spacecraft observation time.

Source: https://l1.umd.edu/ . Vth is sqrt(2 k T / mp), not bulk speed.
No propagation to Earth and no interpolation are performed here.
"""
from io import BytesIO
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import requests

URL = 'https://l1.umd.edu/data/{year}_CELIAS_Proton_Monitor_5min.zip'
TEMPERATURE_FACTOR = 1.67262192369e-27 * 1e6 / (2 * 1.380649e-23)
MIN_SAMPLES = 6  # At least half an hour of five-minute measurements per metric.
MAX_BYTES = 32 * 1024 * 1024
MONTHS = {name: i for i, name in enumerate(
    ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'), 1)}


def empty_frame():
    return pd.DataFrame({'issue_time': pd.Series(dtype='datetime64[ns, UTC]'),
                         **{m: pd.Series(dtype=float) for m in ('v', 'n', 't')}})


def parse_soho(text, year):
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        if len(parts) != 16 or parts[1] not in MONTHS:
            raise ValueError('Unexpected SOHO/CELIAS row format')
        yy, month, day, stamp = parts[:4]
        if int(yy) != year % 100:
            raise ValueError('SOHO archive year mismatch')
        doy, hour, minute, second = map(int, stamp.split(':'))
        at = pd.Timestamp(year=year, month=MONTHS[month], day=int(day),
                          hour=hour, minute=minute, second=second, tz='UTC')
        if at.dayofyear != doy:
            raise ValueError('SOHO day of year mismatch')
        speed, density, thermal = map(float, parts[4:7])
        rows.append((at, speed, density, thermal))
    if not rows:
        raise ValueError('No SOHO/CELIAS observations parsed')
    frame = pd.DataFrame(rows, columns=['issue_time', 'v', 'n', 'vth'])
    if frame.issue_time.duplicated().any():
        raise ValueError('Duplicate SOHO observation timestamps')
    for metric in ('v', 'n', 'vth'):
        frame[metric] = frame[metric].where(np.isfinite(frame[metric]) & frame[metric].gt(0))
    # Convert each observed thermal speed before averaging; mean(vth)**2 is wrong.
    frame['t'] = TEMPERATURE_FACTOR * frame.pop('vth') ** 2
    grouped = frame.set_index('issue_time').sort_index().resample('1h')
    return grouped.mean().where(grouped.count() >= MIN_SAMPLES).reset_index()


class SOHOLoader:
    def __init__(self):
        # Scoped to one adapter/backfill invocation: never freeze a growing archive.
        self._years = {}

    def _year(self, year):
        if year not in self._years:
            with requests.get(URL.format(year=year), stream=True, timeout=(10, 120)) as response:
                if response.status_code == 404:
                    self._years[year] = empty_frame()
                    return self._years[year]
                response.raise_for_status()
                content = bytearray()
                for chunk in response.iter_content(1024 * 1024):
                    content.extend(chunk)
                    if len(content) > MAX_BYTES:
                        raise ValueError('Oversized SOHO archive')
            try:
                archive = ZipFile(BytesIO(content))
            except BadZipFile as exc:
                raise ValueError('Invalid SOHO ZIP archive') from exc
            with archive:
                name = f'{year}_CELIAS_Proton_Monitor_5min.txt'
                info = archive.getinfo(name)
                if info.file_size > MAX_BYTES:
                    raise ValueError('Oversized SOHO text')
                self._years[year] = parse_soho(archive.read(name).decode('ascii'), year)
        return self._years[year]

    def load(self, start, end):
        """Inclusive date/time range, matching the historical wide adapter."""
        frames = [self._year(year) for year in range(start.year, end.year + 1)]
        frame = pd.concat(frames, ignore_index=True) if frames else empty_frame()
        return frame.loc[frame.issue_time.between(start, end)].copy()
