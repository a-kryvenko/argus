"""Live loaders return measurements without database access."""
from datetime import UTC, datetime
from functools import partial

import pandas as pd

from clio.providers import f107, geomagnetic, solar_wind
from clio.ingestion.conversion import wide_to_measurements
from clio.ingestion.adapters import clean_records

COLUMNS = ['metric', 'value', 'observed_at', 'received_at', 'source_product']


def load_wind(kind, start, now):
    return measurements(solar_wind.fetch_records(kind), start, now, f'swpc.rtsw_{kind}')


def load_geomagnetic(metric, start, now):
    return measurements(geomagnetic.fetch_records(metric), start, now, f'swpc.{metric}')


def measurements(dataset, start, now, product):
    received = datetime.now(UTC)
    frame = clean_records(wide_to_measurements(dataset), start,
                          pd.Timestamp(now) + pd.Timedelta(microseconds=1), dataset.columns)
    return frame.drop_duplicates(['metric', 'observed_at'], keep='last').assign(
        received_at=received, source_product=product)


def load_f107(start, now):
    return measurements(f107.fetch_f107(), start, now, 'swpc.f107')


def load_celestrak_f107(start, now):
    from clio.providers.celestrak import fetch_celestrak_f107
    return measurements(fetch_celestrak_f107(), start, now, 'celestrak.f107')


def load_solfsmy_means(start, now):
    from clio.providers.solfsmy import fetch_solfsmy_means
    return measurements(fetch_solfsmy_means(), start, now, 'solfsmy.means')


LIVE_LOADERS = {
    'swpc.rtsw_mag': partial(load_wind, 'mag'),
    'swpc.rtsw_plasma': partial(load_wind, 'plasma'),
    'swpc.kp': partial(load_geomagnetic, 'kp'),
    'swpc.dst': partial(load_geomagnetic, 'dst'),
    'swpc.f107': load_f107,
    'celestrak.f107': load_celestrak_f107,
    'solfsmy.means': load_solfsmy_means,
}
