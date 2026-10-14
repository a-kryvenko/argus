"""Compact numerical NOAA ENLIL L1 profiles, independent of NASA DONKI runs."""
import io
import xml.etree.ElementTree as ET
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
import xarray as xr

BASE_URL = 'https://noaa-wsa-enlil-pds.s3.amazonaws.com/'
MAX_BYTES = 3_000_000
NS = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
PROTON_MASS_KG = 1.67262192369e-27


def download(url, *, params=None):
    with requests.get(url, params=params, timeout=(10, 45), stream=True, allow_redirects=False) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError('Unexpected ENLIL response status')
        chunks, size = [], 0
        for chunk in response.iter_content(65536):
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError('Compact ENLIL response exceeds size limit')
            chunks.append(chunk)
        return b''.join(chunks)


def listing(prefix, delimiter=None):
    params = {'list-type': '2', 'prefix': prefix, 'max-keys': 1000}
    if delimiter:
        params['delimiter'] = delimiter
    try:
        root = ET.fromstring(download(BASE_URL, params=params))
    except ET.ParseError as exc:
        raise ValueError('Invalid ENLIL listing') from exc
    if root.findtext('s:IsTruncated', namespaces=NS) == 'true':
        raise ValueError('ENLIL listing exceeds bounded discovery size')
    prefixes = [p.text for p in root.findall('s:CommonPrefixes/s:Prefix', NS)]
    objects = [dict(key=r.findtext('s:Key', namespaces=NS),
                    size=int(r.findtext('s:Size', namespaces=NS)),
                    last_modified=r.findtext('s:LastModified', namespaces=NS))
               for r in root.findall('s:Contents', NS)]
    return prefixes, objects


def discover(now):
    """Look back three days, at most eight recent run directories; no full3D downloads."""
    runs = []
    for days in range(3):
        prefixes, _ = listing('wsa_enlil.' + (now-timedelta(days=days)).strftime('%Y%m%d'), '/')
        runs.extend(prefixes)
    found = []
    for run in sorted(set(runs), reverse=True)[:8]:
        models, _ = listing(run, '/')
        for model in models[:4]:
            if not model.startswith(run+'wsa_enlil_'):
                continue
            # Some operational versions use the Earth filename for an L1-labelled profile.
            for filename in ('evo.l1.nc', 'evo.earth.nc'):
                _, objects = listing(model+filename)
                matches = [o for o in objects if o['key'] == model+filename and 0 < o['size'] <= MAX_BYTES]
                if matches:
                    found.extend(matches)
                    break
    return found


def parse_profile(raw):
    """Keep physical units explicit; mass density is proton-equivalent, not measured np."""
    if len(raw) > MAX_BYTES:
        raise ValueError('Compact ENLIL profile exceeds size limit')
    engine = 'scipy' if raw.startswith(b'CDF') else 'h5netcdf'
    with xr.open_dataset(io.BytesIO(raw), engine=engine, decode_times=False) as data:
        def attr(name):
            value = data.attrs.get(name)
            return value.decode() if isinstance(value, bytes) else value
        if attr('label') != 'l1' or attr('type') != 'EVO':
            raise ValueError('Expected ENLIL EVO labelled L1')
        for key, unit in [('TIME', 's'), ('D', 'kg/m3'), ('V1', 'm/s'), ('V2', 'm/s'), ('V3', 'm/s')]:
            value = data[key].attrs.get('units')
            if isinstance(value, bytes):
                value = value.decode()
            if value != unit:
                raise ValueError('Unexpected ENLIL units: '+key)
        reference = pd.Timestamp(attr('rundate_cal'), tz='UTC')
        created = pd.Timestamp(attr('creation'), tz='UTC')
        if abs(reference.value/86400e9+40587-float(attr('rundate_mjd'))) > 1e-6:
            raise ValueError('ENLIL calendar/MJD reference mismatch')
        seconds = np.asarray(data['TIME'].values, dtype=float)
        velocity = np.sqrt(sum(np.asarray(data[k].values, dtype=float)**2 for k in ('V1', 'V2', 'V3')))/1000
        density = np.asarray(data['D'].values, dtype=float)/(PROTON_MASS_KG*1e6)
        if seconds.ndim != 1 or not len(seconds) or velocity.shape != seconds.shape or density.shape != seconds.shape:
            raise ValueError('Invalid ENLIL profile dimensions')
        if (not np.isfinite([seconds, velocity, density]).all() or
            (np.diff(seconds) <= 0).any() or (velocity < 0).any() or (density < 0).any()):
            raise ValueError('Invalid ENLIL coordinates or physical values')
        clock = reference + pd.to_timedelta(seconds, unit='s')
        return dict(model_created_at=created.to_pydatetime(), model_version=str(attr('version')),
            target='L1', density_kind='proton_equivalent',
            points=[dict(valid_time=t.isoformat(), speed_km_s=float(v), density_cm3=float(n))
                    for t, v, n in zip(clock, velocity, density)])
