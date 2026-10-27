"""Live loaders return measurements without database access."""
from datetime import UTC, datetime
from functools import partial

import pandas as pd

from clio.providers import f107, geomagnetic_loader, solar_wind_loader
from clio.ingestion.conversion import geomagnetic_measurements, wind_measurements
from clio.ingestion.conversion import wide_to_measurements
from clio.ingestion.adapters import clean_records

COLUMNS = ['metric', 'value', 'observed_at', 'received_at', 'source_product', 'quality']


def load_wind(kind, start, now):
    return pd.DataFrame(wind_measurements(kind, solar_wind_loader.fetch_records(kind)), columns=COLUMNS)


def load_geomagnetic(metric, start, now):
    return pd.DataFrame(geomagnetic_measurements(metric, geomagnetic_loader.fetch_records(metric)), columns=COLUMNS)


def load_f107(start, now):
    wide = f107.fetch_f107()
    received = datetime.now(UTC)
    frame = clean_records(wide_to_measurements(wide), start, pd.Timestamp(now) + pd.Timedelta(microseconds=1), ('f10_7',))
    return frame.drop_duplicates(['metric', 'observed_at'], keep='last').assign(
        received_at=received, source_product='swpc.f107', quality=None)


LIVE_LOADERS = {
    'swpc.rtsw_mag': partial(load_wind, 'mag'),
    'swpc.rtsw_plasma': partial(load_wind, 'plasma'),
    'swpc.kp': partial(load_geomagnetic, 'kp'),
    'swpc.dst': partial(load_geomagnetic, 'dst'),
    'swpc.f107': load_f107,
}
