"""Historical adapters reuse provider parsers without gap filling."""
from datetime import timedelta

import pandas as pd

from clio.ingestion.conversion import wide_to_measurements
from clio.ingestion.products import OBSERVATIONS, PRODUCTS


class WideHistoryAdapter:
    def __init__(self, loader, metrics):
        self.loader = loader
        self.metrics = metrics

    def fetch(self, start, end):
        # Provider requests use inclusive UTC calendar dates.
        frame = self.loader(start.replace(hour=0, minute=0, second=0, microsecond=0),
                            end - timedelta(microseconds=1))
        columns = [column for column in self.metrics if column in frame]
        frame = wide_to_measurements(frame[['issue_time', *columns]])
        return clean_records(frame, start, end, self.metrics)


def clean_records(frame, start, end, metrics):
    frame = frame.copy()
    frame['observed_at'] = pd.to_datetime(frame.observed_at, utc=True)
    frame['value'] = pd.to_numeric(frame.value, errors='coerce')
    valid = pd.Series([row.metric in metrics and OBSERVATIONS[row.metric].accepts(row.value)
                       for row in frame.itertuples(index=False)], index=frame.index, dtype=bool)
    return frame.loc[(frame.observed_at >= start) & (frame.observed_at < end) & valid].copy()


class DailyHistoryAdapter:
    def __init__(self, loader, metrics):
        self.loader = loader
        self.metrics = metrics

    def fetch(self, start, end):
        return clean_records(self.loader(pd.Timestamp(start), pd.Timestamp(end)), start, end, self.metrics)


class CelesTrakHistoryAdapter:
    def __init__(self):
        self.frame = None

    def fetch(self, start, end):
        from clio.providers.celestrak_loader import CELESTRAK_HISTORY_URL, fetch_celestrak_f107, METRICS
        if self.frame is None:
            self.frame = fetch_celestrak_f107(url=CELESTRAK_HISTORY_URL)
        return clean_records(self.frame, start, end, METRICS)


class SOLFSMYHistoryAdapter:
    def __init__(self):
        self.frame = None

    def fetch(self, start, end):
        from clio.providers.solfsmy_loader import fetch_solfsmy_means, MEAN_COLUMNS
        if self.frame is None:
            self.frame = fetch_solfsmy_means()
        return clean_records(self.frame, start, end, MEAN_COLUMNS)


def historical_adapters():
    from clio.providers.omniweb_loader import OMNIWeb_Loader
    from clio.providers.spdf_loader import SPDF_Loader
    from clio.providers.gfz_loader import load_gfz_f107
    from clio.providers.soho_loader import SOHOLoader
    return {
        'omni.hourly': WideHistoryAdapter(OMNIWeb_Loader.load, PRODUCTS['omni.hourly'].metrics),
        'ace.plasma_hourly': WideHistoryAdapter(SPDF_Loader.load_plasma, ('v', 'n', 't')),
        'soho.plasma_hourly': WideHistoryAdapter(SOHOLoader().load, ('v', 'n', 't')),
        'gfz.f107': DailyHistoryAdapter(load_gfz_f107, ('f10_7',)),
        'celestrak.f107_archive': CelesTrakHistoryAdapter(),
        'solfsmy.means_archive': SOLFSMYHistoryAdapter(),
    }
