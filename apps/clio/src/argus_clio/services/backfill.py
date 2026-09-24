"""Restore available historical measurements without replacing collected values."""
import asyncio
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from requests import RequestException
from sqlalchemy import select

from clio.dataloaders.omniweb_loader import OMNIWeb_Loader
from clio.dataloaders.spdf_loader import SPDF_Loader
from clio.observations import wide_to_measurements, REQUIRED_METRICS
from common.config import get_config
from argus_clio.db.models import Measurement
from argus_clio.services.density_history import _download_history
from argus_clio.services.sensor_observations import (
    _upsert_measurements, _upsert_normalized_observations, normalize_measurements,
)

logger = logging.getLogger(__name__)
COLUMNS = ['metric', 'value', 'observed_at']


def load_history(start, end):
    frames, errors = [], []
    # Providers accept inclusive calendar days; clip every result to [start, end).
    last = end - timedelta(microseconds=1)
    def fetch(name, loader):
        try:
            frame = loader()
            if frame.empty:
                raise ValueError('No historical measurements returned')
            frames.append(frame)
        except (RequestException, OSError, ValueError, KeyError, RuntimeError) as exc:
            errors.append(f'{name}: {exc}')
            logger.warning('Backfill source %s unavailable: %s', name, exc)

    fetch('OMNI', lambda: wide_to_measurements(OMNIWeb_Loader.load(start, last)))
    # ACE magnetic fields in the existing loader are GSE, whereas observations
    # require GSM. Only use its plasma channels; never substitute GSE By/Bz.
    def ace():
        frame = SPDF_Loader.load(start, last)
        for metric in ('v', 'n', 't'):
            frame.loc[frame[metric] < 0, metric] = np.nan
        return wide_to_measurements(frame[['issue_time', 'v', 'n', 't']])
    fetch('ACE plasma', ace)
    fetch('solar indices', lambda: _download_history(pd.Timestamp(start), pd.Timestamp(end), get_config()))
    if not frames:
        raise RuntimeError('No backfill sources available: ' + '; '.join(errors))
    frame = pd.concat(frames, ignore_index=True)
    frame.observed_at = pd.to_datetime(frame.observed_at, utc=True)
    frame.value = pd.to_numeric(frame.value, errors='coerce')
    frame = frame.loc[(frame.observed_at >= start) & (frame.observed_at < end)]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=['value', 'observed_at'])
    frame = frame.drop_duplicates(['metric', 'observed_at'], keep='first')
    if frame.empty:
        raise RuntimeError('Sources returned no valid measurements in the requested range')
    return frame[COLUMNS], errors


async def backfill(session, start, end):
    measurements, errors = await asyncio.to_thread(load_history, start, end)
    # Read the merged DB state after insert, so concurrent/pre-existing values win.
    await _upsert_measurements(session, measurements, track_receipt=False, replace_existing=False)
    rows = await session.execute(select(Measurement.metric, Measurement.value, Measurement.observed_at)
        .where(Measurement.observed_at >= start, Measurement.observed_at < end)
        .order_by(Measurement.observed_at))
    stored = pd.DataFrame(rows.all(), columns=COLUMNS)
    normalized = normalize_measurements(stored)
    await _upsert_normalized_observations(session, normalized)
    await session.commit()
    clock = pd.date_range(start, end, freq='h', inclusive='left')
    missing = {}
    for metric in REQUIRED_METRICS:
        valid = stored.loc[(stored.metric == metric) & np.isfinite(stored.value), 'observed_at']
        hours = pd.DatetimeIndex(pd.to_datetime(valid, utc=True)).floor('h').unique()
        missing[metric] = len(clock.difference(hours))
    return {'from': start.isoformat(), 'to_exclusive': end.isoformat(),
            'downloaded_measurements': len(measurements), 'normalized_hours': len(normalized),
            'missing_observed_hours': missing, 'source_errors': errors}
