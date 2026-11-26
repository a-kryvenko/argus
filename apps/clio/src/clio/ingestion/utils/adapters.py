"""Convert and validate provider records; adapt historical loaders."""
from functools import cache, partial

import pandas as pd

from clio.ingestion.utils.products import OBSERVATIONS, PRODUCTS


def wide_to_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    if 'metric' in frame:
        return frame.copy()
    metrics = [metric for metric in frame if metric in OBSERVATIONS and OBSERVATIONS[metric].kind == 'numeric']
    return (frame.melt(id_vars='timestamp', value_vars=metrics, var_name='metric', value_name='value')
            .rename(columns={'timestamp': 'observed_at'}).dropna(subset=['observed_at', 'value']))


class WideHistoryAdapter:
    def __init__(self, loader, metrics):
        self.loader = loader
        self.metrics = metrics

    def fetch(self, start, end):
        frame = self.loader(start, end)
        return clean_records(wide_to_measurements(frame), start, end, self.metrics)


def clean_records(frame, start, end, metrics):
    frame = frame.copy()
    frame['observed_at'] = pd.to_datetime(frame.observed_at, utc=True)
    frame['value'] = pd.to_numeric(frame.value, errors='coerce')
    def accepts(row):
        if row.metric not in metrics:
            return False
        if getattr(row, 'quality', None) in ('flagged', 'missing'):
            return pd.isna(row.value)
        return OBSERVATIONS[row.metric].accepts(row.value)

    valid = pd.Series([accepts(row) for row in frame.itertuples(index=False)],
                      index=frame.index, dtype=bool)
    return frame.loc[(frame.observed_at >= start) & (frame.observed_at < end) & valid].copy()


class CelesTrakHistoryAdapter:
    def __init__(self):
        self.frame = None

    def fetch(self, start, end):
        from clio.providers.celestrak import CELESTRAK_HISTORY_URL, fetch_celestrak_f107, METRICS
        if self.frame is None:
            self.frame = fetch_celestrak_f107(url=CELESTRAK_HISTORY_URL)
        return clean_records(wide_to_measurements(self.frame), start, end, METRICS)


class SOLFSMYHistoryAdapter:
    def __init__(self):
        self.frame = None

    def fetch(self, start, end):
        from clio.providers.solfsmy import fetch_solfsmy_means, MEAN_COLUMNS
        if self.frame is None:
            self.frame = fetch_solfsmy_means()
        return clean_records(wide_to_measurements(self.frame), start, end, MEAN_COLUMNS)


def _goes_flux_history(product):
    from clio.providers.goes_flux import fetch_records
    # One rolling snapshot covers all missing day ranges in this backfill.
    snapshot = cache(partial(fetch_records, product, historical=True))
    return WideHistoryAdapter(lambda start, end: snapshot(), PRODUCTS[product].metrics)


def historical_adapters():
    from clio.providers.omniweb import OMNIWeb_Loader
    from clio.providers.spdf import SPDF_Loader
    from clio.providers.gfz import load_gfz_f107
    from clio.providers.soho import SOHOLoader
    return {
        'omni.hourly': WideHistoryAdapter(OMNIWeb_Loader.load, PRODUCTS['omni.hourly'].metrics),
        'ace.plasma_hourly': WideHistoryAdapter(SPDF_Loader.load_plasma, ('v', 'n', 't')),
        'soho.plasma_hourly': WideHistoryAdapter(SOHOLoader().load, ('v', 'n', 't')),
        'gfz.f107': WideHistoryAdapter(load_gfz_f107, ('f10_7',)),
        'celestrak.f107_archive': CelesTrakHistoryAdapter(),
        'solfsmy.means_archive': SOLFSMYHistoryAdapter(),
        'goes.xrs': _goes_flux_history('goes.xrs'),
        'goes.protons': _goes_flux_history('goes.protons'),
    }
